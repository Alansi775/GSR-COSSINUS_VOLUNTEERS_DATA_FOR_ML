"""
Per-volunteer (baseline-referenced) normalization + sliding-window
extraction, now producing TWO things per window:

  1. seq window  [window_size, 3]  -> GSR(z), RR(z), GSR(z) first-difference
     (fed to the CNN+LSTM branch)
  2. scalar feature vector [5]     -> RMSSD, SDNN, GSR slope, GSR range,
     GSR peak count (fed to a small MLP branch)

Both branches use the volunteer's baseline-referenced z-scored signal, so
a resistance-fixed file (35-144) and a native-conductance file (V145+) end
up on the same relative scale even though their absolute units differ.
"""
import numpy as np
import config as cfg


def per_volunteer_normalize(df):
    """
    Z-score Conductance and RR using stats computed ONLY from this
    volunteer's baseline stages (Calibration + Normal), not the whole
    session. This avoids the stress-phase variance diluting the very
    signal we want the model to learn.
    """
    df = df.copy()
    baseline_mask = df[cfg.COL_STAGE].isin(cfg.BASELINE_STAGES)

    for col in [cfg.COL_COND, cfg.COL_RR]:
        baseline_vals = df.loc[baseline_mask, col].dropna()
        if len(baseline_vals) < 5:
            mu = df[col].mean()
            sigma = df[col].std()
        else:
            mu = baseline_vals.mean()
            sigma = baseline_vals.std()

        if not np.isfinite(sigma) or sigma == 0:
            sigma = 1.0
        if not np.isfinite(mu):
            mu = 0.0

        df[col + "_z"] = (df[col] - mu) / sigma

    return df


def _rmssd(rr_raw):
    diffs = np.diff(rr_raw)
    if len(diffs) == 0:
        return 0.0
    return float(np.sqrt(np.mean(diffs ** 2)))


def _sdnn(rr_raw):
    return float(np.std(rr_raw))


def _slope(cond_raw):
    t = np.arange(len(cond_raw))
    if len(t) < 2 or np.std(cond_raw) == 0:
        return 0.0
    slope = np.polyfit(t, cond_raw, 1)[0]
    return float(slope)


def _range(cond_raw):
    return float(np.max(cond_raw) - np.min(cond_raw))


def _peak_count(cond_raw):
    if len(cond_raw) < 3:
        return 0.0
    is_peak = (cond_raw[1:-1] > cond_raw[:-2]) & (cond_raw[1:-1] > cond_raw[2:])
    return float(np.sum(is_peak))


def compute_window_scalar_features(cond_z_win, rr_raw_win):
    """
    Returns np.array of shape [5]: [rmssd, sdnn, slope, range, peak_count].

    RMSSD/SDNN are computed from RAW RR (ms) — RR was never affected by the
    resistance/conductance mislabeling, so raw ms is consistent and
    physiologically meaningful across every volunteer, and z-scoring it
    first would partially re-measure the normalization instead of adding
    new information.

    GSR slope/range/peak-count are computed from the per-volunteer
    Z-SCORED conductance, not raw — because raw values are NOT on a
    consistent scale across volunteers (files 35-144 store 1/resistance,
    files V145+ store native conductance, and the two aren't calibrated
    to the same absolute units). Z-scoring puts both file types on the
    same relative scale before these shape-based features are computed.
    """
    return np.array([
        _rmssd(rr_raw_win),
        _sdnn(rr_raw_win),
        _slope(cond_z_win),
        _range(cond_z_win),
        _peak_count(cond_z_win),
    ], dtype=np.float32)


def extract_stage_windows(df, stage_name, label, window_size, step_size):
    """
    Sliding windows of length window_size, step step_size, taken only from
    contiguous runs of rows where Stage == stage_name (never crosses a
    stage boundary).
    Returns:
        seq_windows: list of np.ndarray [window_size, 3]  (GSR_z, RR_z, dGSR_z)
        feat_windows: list of np.ndarray [5]              (engineered scalars)
        labels: list of int
    """
    seq_windows, feat_windows = [], []
    stage_mask = (df[cfg.COL_STAGE] == stage_name).values
    n = len(df)
    cond_z = df[cfg.COL_COND + "_z"].values
    rr_z = df[cfg.COL_RR + "_z"].values
    rr_raw = df[cfg.COL_RR].values

    i = 0
    while i < n:
        if not stage_mask[i]:
            i += 1
            continue
        j = i
        while j < n and stage_mask[j]:
            j += 1
        block_len = j - i
        start = 0
        while start + window_size <= block_len:
            c_win = cond_z[i + start:i + start + window_size]
            r_win = rr_z[i + start:i + start + window_size]
            if not (np.isnan(c_win).any() or np.isnan(r_win).any()):
                d_win = np.diff(c_win, prepend=c_win[0])
                seq = np.stack([c_win, r_win, d_win], axis=-1)
                r_raw_win = rr_raw[i + start:i + start + window_size]
                feat = compute_window_scalar_features(c_win, r_raw_win)
                seq_windows.append(seq)
                feat_windows.append(feat)
            start += step_size
        i = j

    labels = [label] * len(seq_windows)
    return seq_windows, feat_windows, labels


def build_dataset(volunteer_data, window_size=cfg.WINDOW_SIZE, step_size=cfg.STEP_SIZE):
    """
    volunteer_data: dict {volunteer_key: raw DataFrame}
    Returns:
        X_seq:  np.ndarray [N, window_size, 3] float32
        X_feat: np.ndarray [N, 5] float32
        y:      np.ndarray [N] int64
        groups: np.ndarray [N] str
    """
    X_seq, X_feat, y, groups = [], [], [], []

    for key, df in volunteer_data.items():
        df_norm = per_volunteer_normalize(df)

        r_seq, r_feat, r_y = extract_stage_windows(
            df_norm, cfg.STAGE_RELAXED, label=0,
            window_size=window_size, step_size=step_size)
        s_seq, s_feat, s_y = extract_stage_windows(
            df_norm, cfg.STAGE_STRESSED, label=1,
            window_size=window_size, step_size=step_size)

        n_new = len(r_seq) + len(s_seq)
        if n_new == 0:
            continue

        X_seq.extend(r_seq + s_seq)
        X_feat.extend(r_feat + s_feat)
        y.extend(r_y + s_y)
        groups.extend([key] * n_new)

    if not X_seq:
        raise RuntimeError("No windows were extracted from any volunteer. "
                            "Check stage names / window size vs stage duration.")

    X_seq = np.asarray(X_seq, dtype=np.float32)
    X_feat = np.asarray(X_feat, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups)
    return X_seq, X_feat, y, groups