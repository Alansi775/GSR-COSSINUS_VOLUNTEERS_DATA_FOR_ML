"""
Run the trained model on a specific volunteer's data and print a
per-window prediction timeline — this is for validating the model
against something you know is true (e.g. a moment you saw on video
where a volunteer visibly got stressed), before treating the model as
production-ready.

Usage:
    python predict.py --volunteer 57
    python predict.py --volunteer 57 --start 180 --end 240

--start / --end are in seconds, matching the Time_sec column, and are
optional — omit either to run from the beginning / to the end of the
session.
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F

import config as cfg
from data_loader import discover_volunteer_files, load_volunteer_csv
from feature_extraction import per_volunteer_normalize, compute_window_scalar_features
from model import CNNLSTMHybrid


def load_normalization():
    seq_mean = np.load(cfg.GLOBAL_MEAN_PATH)
    seq_std = np.load(cfg.GLOBAL_STD_PATH)
    feat_mean = np.load(cfg.GLOBAL_FEAT_MEAN_PATH)
    feat_std = np.load(cfg.GLOBAL_FEAT_STD_PATH)
    return seq_mean, seq_std, feat_mean, feat_std


def load_model(device):
    model = CNNLSTMHybrid(n_channels=cfg.N_CHANNELS,
                           n_scalar_features=cfg.N_SCALAR_FEATURES).to(device)
    model.load_state_dict(torch.load(cfg.BEST_MODEL_PATH, map_location=device))
    model.eval()
    return model


def run_inference(volunteer_key, start_sec=None, end_sec=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_map = discover_volunteer_files(cfg.DATA_ROOT)
    if volunteer_key not in file_map:
        available = ", ".join(sorted(file_map.keys())[:15]) + ", ..."
        raise ValueError(
            f"Volunteer '{volunteer_key}' not found under {cfg.DATA_ROOT}. "
            f"Some available keys: {available}"
        )

    df = load_volunteer_csv(file_map[volunteer_key], volunteer_key)
    df_norm = per_volunteer_normalize(df)

    if start_sec is not None:
        df_norm = df_norm[df_norm[cfg.COL_TIME] >= start_sec]
    if end_sec is not None:
        df_norm = df_norm[df_norm[cfg.COL_TIME] <= end_sec]
    df_norm = df_norm.reset_index(drop=True)

    if len(df_norm) < cfg.WINDOW_SIZE:
        raise ValueError(
            f"Only {len(df_norm)} rows in the requested range — need at least "
            f"{cfg.WINDOW_SIZE} (window size). Widen --start/--end."
        )

    seq_mean, seq_std, feat_mean, feat_std = load_normalization()
    model = load_model(device)

    cond_z = df_norm[cfg.COL_COND + "_z"].values
    rr_z = df_norm[cfg.COL_RR + "_z"].values
    rr_raw = df_norm[cfg.COL_RR].values
    time_vals = df_norm[cfg.COL_TIME].values
    stage_vals = df_norm[cfg.COL_STAGE].values

    window_size = cfg.WINDOW_SIZE
    step_size = cfg.STEP_SIZE
    n = len(df_norm)

    results = []
    start = 0
    while start + window_size <= n:
        c_win = cond_z[start:start + window_size]
        r_win = rr_z[start:start + window_size]

        if not (np.isnan(c_win).any() or np.isnan(r_win).any()):
            d_win = np.diff(c_win, prepend=c_win[0])
            seq = np.stack([c_win, r_win, d_win], axis=-1)
            r_raw_win = rr_raw[start:start + window_size]
            feat = compute_window_scalar_features(c_win, r_raw_win)

            seq_n = (seq - seq_mean) / seq_std
            feat_n = (feat - feat_mean) / feat_std

            seq_t = torch.from_numpy(seq_n).float().unsqueeze(0).to(device)
            feat_t = torch.from_numpy(feat_n).float().unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(seq_t, feat_t)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
            label = "Stressed" if pred == 1 else "Relaxed"

            t_start = time_vals[start]
            t_end = time_vals[start + window_size - 1]
            true_stage = stage_vals[start + window_size // 2]

            results.append({
                "t_start": t_start, "t_end": t_end,
                "prediction": label, "confidence": probs[pred],
                "true_stage": true_stage,
            })
        start += step_size

    if not results:
        print("No valid windows in this range (likely NaN gaps). Try a wider range.")
        return []

    print(f"\nVolunteer: {volunteer_key}   Windows: {len(results)}")
    print(f"{'Time range':>16} | {'Prediction':>10} | {'Confidence':>10} | Logged stage (ground truth)")
    print("-" * 75)
    for r in results:
        marker = ""
        if r["true_stage"] in (cfg.STAGE_RELAXED, cfg.STAGE_STRESSED):
            expected = "Relaxed" if r["true_stage"] == cfg.STAGE_RELAXED else "Stressed"
            marker = "  OK" if expected == r["prediction"] else "  MISMATCH"
        print(f"{r['t_start']:>6.0f}-{r['t_end']:<6.0f}s | {r['prediction']:>10} | "
              f"{r['confidence']*100:>8.1f}% | {r['true_stage']}{marker}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the trained model on one volunteer's data.")
    parser.add_argument("--volunteer", required=True, help="Volunteer key, e.g. 57 or V146")
    parser.add_argument("--start", type=float, default=None, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=None, help="End time in seconds")
    args = parser.parse_args()

    run_inference(args.volunteer, args.start, args.end)