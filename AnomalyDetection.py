"""
MLSP PBL: Anomaly detection in non-stationary ECG signals using Isolation Forest

Dataset : MIT-BIH Arrhythmia Database (PhysioNet, https://physionet.org/content/mitdb/1.0.0/)
Signal processing : (1) zero-phase Butterworth bandpass filter, (2) wavelet (db4) band-energy
                    features, plus time-domain shape features and RR-interval features
ML models : Isolation Forest, One-Class SVM, Local Outlier Factor
Research question : how should features be normalized when the signal is non-stationary
                    across patients and over time?  (global vs calibration vs sliding window)

Usage examples
    python mlsp_pbl_final.py --plots
    python mlsp_pbl_final.py --experiments features
    python mlsp_pbl_final.py --experiments normalization
    python mlsp_pbl_final.py --experiments models
    python mlsp_pbl_final.py --demo 119 --model ocsvm

The dataset is read from the folder "mitdb_local" if it exists, otherwise it is streamed
from PhysioNet. Beats are cached in beats_cache_v2.npz after the first run.
"""
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")   # silences a joblib warning on Windows

import argparse
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
import pywt
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import skew, kurtosis
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_fscore_support)


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
FS = 360                      # MIT-BIH sampling rate (Hz)
LOCAL_DIR = "mitdb_local"     # local copy of the dataset (optional)
CACHE_FILE = "beats_cache_v2.npz"

PRE = int(0.25 * FS)          # samples kept before the R peak
POST = int(0.40 * FS)         # samples kept after the R peak
NORMAL = set("NLRej")         # beat symbols treated as normal
ABNORMAL = set("AaJSVFE")     # beat symbols treated as abnormal
EXCLUDE = {"102", "104", "107", "217"}   # paced records, left out

ALL_RECORDS = [str(i) for i in (
    list(range(100, 110)) + list(range(111, 120)) + list(range(121, 125)) +
    [200, 201, 202, 203, 205] + list(range(207, 211)) +
    [212, 213, 214, 215, 217] + list(range(219, 224)) +
    [228] + list(range(230, 235)))]
RECORDS = [r for r in ALL_RECORDS if r not in EXCLUDE]


# ---------------------------------------------------------------------------
# 2. Data loading
# ---------------------------------------------------------------------------
def _read(rec_id):
    """Read a record and its annotations, with retries for PhysioNet server errors."""
    for attempt in range(4):
        try:
            if os.path.isdir(LOCAL_DIR):
                path = os.path.join(LOCAL_DIR, rec_id)
                return wfdb.rdrecord(path), wfdb.rdann(path, "atr")
            return (wfdb.rdrecord(rec_id, pn_dir="mitdb"),
                    wfdb.rdann(rec_id, "atr", pn_dir="mitdb"))
        except Exception as e:
            print(f"  record {rec_id}: attempt {attempt + 1} failed ({type(e).__name__})")
            time.sleep(3)
    raise RuntimeError(f"Could not load record {rec_id}")


def load_record(rec_id):
    """Return the MLII lead (picked by name, record 114 has its leads swapped) and annotations."""
    rec, ann = _read(rec_id)
    ch = rec.sig_name.index("MLII")
    return rec.p_signal[:, ch], ann


# ---------------------------------------------------------------------------
# 3. Signal processing technique 1: bandpass filtering
# ---------------------------------------------------------------------------
def bandpass_ecg(sig, low=0.5, high=40.0, fs=FS, order=4):
    """Zero-phase Butterworth bandpass. Removes baseline wander and high-frequency noise
    while keeping R peaks at their original positions."""
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, sig)


# ---------------------------------------------------------------------------
# 4. Beat segmentation and dataset building
# ---------------------------------------------------------------------------
def extract_beats(rec_id):
    """Filter one record, then cut a fixed window around every labeled beat."""
    sig, ann = load_record(rec_id)
    filt = bandpass_ecg(sig)
    beats, labels, positions = [], [], []
    for s, sym in zip(ann.sample, ann.symbol):
        if sym in NORMAL:
            lab = 0
        elif sym in ABNORMAL:
            lab = 1
        else:
            continue
        if s - PRE < 0 or s + POST > len(filt):
            continue
        beats.append(filt[s - PRE:s + POST])
        labels.append(lab)
        positions.append(s)
    return np.array(beats), np.array(labels), np.array(positions)


def build_dataset(records=RECORDS, cache=CACHE_FILE):
    """Return X (beats), y (0 normal / 1 abnormal), rec (record id per beat),
    pos (R-peak sample index). Cached after the first run."""
    if os.path.exists(cache):
        d = np.load(cache)
        return d["X"], d["y"], d["rec"], d["pos"]
    X, y, rec, pos = [], [], [], []
    for r in records:
        print("Processing", r)
        b, l, p = extract_beats(r)
        X.append(b)
        y.append(l)
        pos.append(p)
        rec.extend([r] * len(l))
    X, y = np.vstack(X), np.concatenate(y)
    rec, pos = np.array(rec), np.concatenate(pos)
    np.savez(cache, X=X, y=y, rec=rec, pos=pos)
    return X, y, rec, pos


# ---------------------------------------------------------------------------
# 5. Signal processing technique 2: feature extraction
# ---------------------------------------------------------------------------
def time_features(X):
    """8 shape statistics per beat."""
    return np.column_stack([
        X.mean(axis=1), X.std(axis=1), skew(X, axis=1), kurtosis(X, axis=1),
        np.ptp(X, axis=1), X.max(axis=1), X.min(axis=1), (X ** 2).mean(axis=1)])


def wavelet_features(X, wavelet="db4", level=4):
    """Relative energy per wavelet band (5 values) plus log total energy."""
    out = []
    for beat in X:
        coeffs = pywt.wavedec(beat, wavelet, level=level)
        e = np.array([np.sum(c ** 2) for c in coeffs])
        total = e.sum() + 1e-12
        out.append(np.r_[e / total, np.log10(total)])
    return np.array(out)


def rr_features(pos, rec):
    """Previous RR interval, next RR interval, and previous RR relative to the
    mean of the preceding 8 intervals (all per patient)."""
    out = np.zeros((len(pos), 3))
    for r in np.unique(rec):
        idx = np.where(rec == r)[0]
        rr = np.diff(pos[idx].astype(float)) / FS
        prev = np.r_[rr[0], rr]
        nxt = np.r_[rr, rr[-1]]
        loc = np.array([prev[max(0, i - 8):i].mean() if i > 0 else prev[0]
                        for i in range(len(prev))])
        out[idx] = np.column_stack([prev, nxt, prev / loc])
    return out


def all_features(X, rec, pos):
    """Return dict with the three feature blocks (time 8, wavelet 6, RR 3)."""
    return {"time": time_features(X),
            "wavelet": wavelet_features(X),
            "rr": rr_features(pos, rec)}


# ---------------------------------------------------------------------------
# 6. Adaptive (per-patient) normalization, the research contribution
# ---------------------------------------------------------------------------
def adaptive_normalize(F, rec, mode="sliding", window=300, n_cal=300):
    """Robust per-patient normalization using median and IQR. Uses no labels.
    global      : no change (a single scaler is applied later for everyone)
    calibration : statistics from the patient's first n_cal beats, fixed afterwards
    sliding     : statistics from the previous `window` beats only (causal)"""
    if mode == "global":
        return F
    out = np.zeros_like(F, dtype=float)
    for r in np.unique(rec):
        idx = np.where(rec == r)[0]
        df = pd.DataFrame(F[idx])
        if mode == "calibration":
            ref = df.iloc[:n_cal]
            med = ref.median()
            iqr = ref.quantile(0.75) - ref.quantile(0.25)
            z = (df - med) / (iqr + 1e-9)
        elif mode == "sliding":
            roll = df.shift(1).rolling(window, min_periods=30)
            med = roll.median()
            iqr = roll.quantile(0.75) - roll.quantile(0.25)
            z = (df - med) / (iqr + 1e-9)
            ref = df.iloc[:30]                       # fallback for the first beats
            m0 = ref.median()
            i0 = ref.quantile(0.75) - ref.quantile(0.25)
            z = z.where(~z.isna(), (df - m0) / (i0 + 1e-9))
        else:
            raise ValueError(mode)
        out[idx] = np.clip(z.values, -20, 20)
    return out


def hybrid_features(X, rec, pos):
    """Best recipe found: time features with sliding normalization, RR left raw."""
    f = all_features(X, rec, pos)
    return np.hstack([adaptive_normalize(f["time"], rec, "sliding"), f["rr"]])


# ---------------------------------------------------------------------------
# 7. Models and evaluation
# ---------------------------------------------------------------------------
def make_iforest():
    return IsolationForest(n_estimators=200, random_state=0, n_jobs=-1)


def make_ocsvm():
    return OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)


def make_lof():
    return LocalOutlierFactor(n_neighbors=35, novelty=True, n_jobs=-1)


MODELS = {"iforest": make_iforest, "ocsvm": make_ocsvm, "lof": make_lof}


def cross_validate_detector(make_model, F, y, rec, n_splits=5, q=95,
                            max_train=15000, n_cal=5000, seed=0):
    """Patient-wise cross-validation (test patients are never seen in training).
    The model is fit on normal beats only. The decision threshold is the q-th percentile
    of scores on other normal training beats, so no test labels are used."""
    rng = np.random.default_rng(seed)
    rows = []
    for tr, te in GroupKFold(n_splits=n_splits).split(F, y, groups=rec):
        tr_norm = rng.permutation(tr[y[tr] == 0])
        fit_idx = tr_norm[:max_train]
        cal_idx = tr_norm[max_train:max_train + n_cal]

        scaler = StandardScaler().fit(F[fit_idx])
        model = make_model().fit(scaler.transform(F[fit_idx]))

        s_cal = -model.score_samples(scaler.transform(F[cal_idx]))   # higher = more anomalous
        s_test = -model.score_samples(scaler.transform(F[te]))
        pred = (s_test > np.percentile(s_cal, q)).astype(int)

        p, r, f1, _ = precision_recall_fscore_support(
            y[te], pred, average="binary", zero_division=0)
        rows.append({"auc": roc_auc_score(y[te], s_test),
                     "ap": average_precision_score(y[te], s_test),
                     "precision": p, "recall": r, "f1": f1})
    return rows


def _summarize(rows):
    d = pd.DataFrame(rows)
    return {**d.mean().round(3).to_dict(), "f1_std": round(d["f1"].std(), 3)}


# ---------------------------------------------------------------------------
# 8. Experiments
# ---------------------------------------------------------------------------
def exp_features():
    """Isolation Forest with global scaling on different feature sets."""
    X, y, rec, pos = build_dataset()
    f = all_features(X, rec, pos)
    t, w, rr = f["time"], f["wavelet"], f["rr"]
    sets = {"time only": t, "wavelet only": w, "RR only": rr,
            "time+wavelet": np.hstack([t, w]), "time+RR": np.hstack([t, rr]),
            "wavelet+RR": np.hstack([w, rr]), "time+wavelet+RR": np.hstack([t, w, rr])}
    rows = []
    for name, F in sets.items():
        print("Running", name)
        rows.append({"features": name,
                     **_summarize(cross_validate_detector(make_iforest, F, y, rec))})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv("results_features.csv", index=False)


def exp_normalization():
    """Global vs calibration vs sliding normalization (Isolation Forest)."""
    X, y, rec, pos = build_dataset()
    f = all_features(X, rec, pos)
    t, rr = f["time"], f["rr"]
    sets = {"RR only": rr, "time only": t, "time+RR": np.hstack([t, rr]),
            "time+wavelet+RR": np.hstack([t, f["wavelet"], rr])}
    rows = []
    for name, F in sets.items():
        for mode in ["global", "calibration", "sliding"]:
            print("Running", name, mode)
            Fn = adaptive_normalize(F, rec, mode)
            rows.append({"features": name, "normalization": mode,
                         **_summarize(cross_validate_detector(make_iforest, Fn, y, rec))})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv("results_normalization.csv", index=False)


def exp_models():
    """Three models on several feature sets, shape features normalized with the sliding window."""
    X, y, rec, pos = build_dataset()
    f = all_features(X, rec, pos)
    t, w, rr = f["time"], f["wavelet"], f["rr"]

    def slide(F):
        return adaptive_normalize(F, rec, "sliding")

    sets = {"RR only (global)": rr,
            "time (sliding)": slide(t),
            "time+RR (sliding)": slide(np.hstack([t, rr])),
            "time (sliding) + RR raw": np.hstack([slide(t), rr]),
            "time+wavelet (sliding) + RR raw": np.hstack([slide(np.hstack([t, w])), rr])}
    rows = []
    for fname, F in sets.items():
        for mname, mk in MODELS.items():
            print("Running", fname, "|", mname)
            rows.append({"features": fname, "model": mname,
                         **_summarize(cross_validate_detector(mk, F, y, rec))})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv("results_models.csv", index=False)


# ---------------------------------------------------------------------------
# 9. Figures for the report
# ---------------------------------------------------------------------------
def plot_filtering(rec_id="119", seconds=10):
    sig, _ = load_record(rec_id)
    filt = bandpass_ecg(sig)
    n = seconds * FS
    t = np.arange(n) / FS
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(t, sig[:n], lw=0.8)
    ax[0].set_title(f"Raw ECG, record {rec_id}")
    ax[1].plot(t, filt[:n], lw=0.8, color="green")
    ax[1].set_title("After 0.5-40 Hz bandpass")
    ax[1].set_xlabel("Time (s)")
    for a in ax:
        a.set_ylabel("mV")
    plt.tight_layout()

    f1, p1 = welch(sig, fs=FS, nperseg=2048)
    f2, p2 = welch(filt, fs=FS, nperseg=2048)
    plt.figure(figsize=(10, 4))
    plt.semilogy(f1, p1, label="raw")
    plt.semilogy(f2, p2, label="filtered")
    plt.xlim(0, 100)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power spectral density")
    plt.title("Spectrum before and after filtering")
    plt.legend()
    plt.tight_layout()


def plot_mean_beats():
    X, y, rec, pos = build_dataset()
    t = (np.arange(PRE + POST) - PRE) / FS
    plt.figure(figsize=(8, 4))
    plt.plot(t, X[y == 0].mean(axis=0), label="normal (mean)")
    plt.plot(t, X[y == 1].mean(axis=0), label="abnormal (mean)")
    plt.axvline(0, color="gray", ls=":")
    plt.xlabel("Time from R peak (s)")
    plt.ylabel("mV")
    plt.title("Average beat shape, all patients")
    plt.legend()
    plt.tight_layout()


# ---------------------------------------------------------------------------
# 10. Demonstration prototype
# ---------------------------------------------------------------------------
def detect(record_id, model_name="ocsvm", t_start=0, t_len=20,
           q=95, max_train=15000, n_cal=5000, seed=0):
    """Train on normal beats of all OTHER patients, flag abnormal beats in one record,
    print a summary and plot a time window with the detections."""
    X, y, rec, pos = build_dataset()
    F = hybrid_features(X, rec, pos)

    rng = np.random.default_rng(seed)
    test = rec == record_id
    pool = rng.permutation(np.where((~test) & (y == 0))[0])
    fit_idx, cal_idx = pool[:max_train], pool[max_train:max_train + n_cal]

    scaler = StandardScaler().fit(F[fit_idx])
    model = MODELS[model_name]().fit(scaler.transform(F[fit_idx]))
    thr = np.percentile(-model.score_samples(scaler.transform(F[cal_idx])), q)
    pred = (-model.score_samples(scaler.transform(F[test])) > thr).astype(int)

    y_te, p_te = y[test], pos[test]
    tp = int(((pred == 1) & (y_te == 1)).sum())
    fp = int(((pred == 1) & (y_te == 0)).sum())
    fn = int(((pred == 0) & (y_te == 1)).sum())
    print(f"Record {record_id} | model: {model_name}")
    print(f"Beats: {len(y_te)} | truly abnormal: {int(y_te.sum())} | flagged: {int(pred.sum())}")
    print(f"Caught: {tp} | missed: {fn} | false alarms: {fp}")
    if tp + fp > 0:
        print(f"Precision: {tp / (tp + fp):.2f}")
    if tp + fn > 0:
        print(f"Recall: {tp / (tp + fn):.2f}")

    sig, _ = load_record(record_id)
    filt = bandpass_ecg(sig)
    a, b = int(t_start * FS), int((t_start + t_len) * FS)
    t = np.arange(a, b) / FS
    win = (p_te >= a) & (p_te < b)
    true_pk = p_te[win & (y_te == 1)]
    flag_pk = p_te[win & (pred == 1)]

    plt.figure(figsize=(13, 4))
    plt.plot(t, filt[a:b], lw=0.8, color="steelblue")
    plt.scatter(true_pk / FS, filt[true_pk], s=160, facecolors="none",
                edgecolors="black", label="truly abnormal (cardiologist)")
    plt.scatter(flag_pk / FS, filt[flag_pk], s=30, color="red",
                label=f"flagged by {model_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("mV")
    plt.title(f"Record {record_id}: detected anomalies ({t_start} to {t_start + t_len} s)")
    plt.legend(loc="upper right")
    plt.tight_layout()


# ---------------------------------------------------------------------------
# 11. Command line entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Anomaly detection in non-stationary ECG")
    ap.add_argument("--plots", action="store_true", help="report figures (filtering, mean beats)")
    ap.add_argument("--experiments", choices=["features", "normalization", "models", "all"])
    ap.add_argument("--demo", metavar="RECORD", help="run the detector on one record, e.g. 119")
    ap.add_argument("--model", default="ocsvm", choices=list(MODELS))
    ap.add_argument("--t_start", type=float, default=0, help="demo window start (s)")
    args = ap.parse_args()

    if not (args.plots or args.experiments or args.demo):
        # plain "python file.py": show the figures and the demo (fast, about a minute)
        print("No options given: showing the figures and the demo on record 119.")
        print("Use --experiments features | normalization | models to regenerate the tables.\n")
        args.plots = True
        args.demo = "119"
    if args.plots:
        plot_filtering()
        plot_mean_beats()
    if args.experiments in ("features", "all"):
        exp_features()
    if args.experiments in ("normalization", "all"):
        exp_normalization()
    if args.experiments in ("models", "all"):
        exp_models()
    if args.demo:
        detect(args.demo, args.model, t_start=args.t_start)
    plt.show()


if __name__ == "__main__":
    main()