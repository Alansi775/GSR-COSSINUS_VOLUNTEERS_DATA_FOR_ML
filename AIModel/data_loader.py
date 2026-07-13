"""
Loads per-volunteer CSVs, applies the resistance->conductance fix for
files 35-144, and drops excluded/broken volunteers.
"""
import os
import re
import glob
import numpy as np
import pandas as pd

import config as cfg

# Matches things like: cleaned_data_35.csv, 35.csv, cleaned_data_V147f.csv, V147f.csv
_ID_RE = re.compile(r'(V?)(\d+)(f?)(?=\.csv$)', re.IGNORECASE)


def discover_volunteer_files(data_root):
    """
    Recursively scans data_root for CSVs and maps a volunteer key
    (e.g. '35' or 'V147f') to its file path. If multiple files match the
    same key, the first one found wins (a warning is printed).
    """
    pattern = os.path.join(data_root, "**", "*.csv")
    candidates = sorted(glob.glob(pattern, recursive=True))

    file_map = {}
    for path in candidates:
        fname = os.path.basename(path)
        m = _ID_RE.search(fname)
        if not m:
            continue
        v_prefix, number, f_suffix = m.groups()
        key = f"{v_prefix.upper()}{number}{f_suffix.lower()}"
        if key in file_map:
            print(f"[warn] duplicate volunteer key '{key}': "
                  f"keeping {file_map[key]}, ignoring {path}")
            continue
        file_map[key] = path
    return file_map


def volunteer_numeric_id(key):
    """'V147f' -> 147, '35' -> 35"""
    m = re.search(r'(\d+)', key)
    return int(m.group(1)) if m else None


def is_v_prefixed(key):
    return key.upper().startswith('V')


def needs_resistance_fix(key):
    """True if this volunteer's file has resistance mislabeled as conductance."""
    if is_v_prefixed(key):
        return False
    num = volunteer_numeric_id(key)
    return num is not None and cfg.RESISTANCE_ID_MIN <= num <= cfg.RESISTANCE_ID_MAX


def load_volunteer_csv(path, key):
    df = pd.read_csv(path)

    required_cols = {cfg.COL_TIME, cfg.COL_COND, cfg.COL_STAGE, cfg.COL_RR}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"missing columns {missing}")

    df = df.copy()
    df[cfg.COL_COND] = pd.to_numeric(df[cfg.COL_COND], errors="coerce")
    df[cfg.COL_RR] = pd.to_numeric(df[cfg.COL_RR], errors="coerce")

    if needs_resistance_fix(key):
        resistance = df[cfg.COL_COND].replace(0, np.nan)
        conductance = 1.0 / resistance
        df[cfg.COL_COND] = conductance.ffill().bfill()

    return df


def load_all_volunteers(data_root=None, verbose=True):
    """
    Returns:
        data: dict {volunteer_key: DataFrame}
        skipped: list of (key, reason) tuples
    """
    data_root = data_root or cfg.DATA_ROOT
    file_map = discover_volunteer_files(data_root)

    if not file_map:
        raise FileNotFoundError(
            f"No CSV files matched under '{data_root}'. Check config.DATA_ROOT."
        )

    data = {}
    skipped = []

    for key, path in sorted(file_map.items(),
                             key=lambda kv: (volunteer_numeric_id(kv[0]) or -1)):
        num = volunteer_numeric_id(key)
        if num in cfg.EXCLUDED_VOLUNTEERS:
            skipped.append((key, "excluded (known bad data)"))
            continue
        try:
            df = load_volunteer_csv(path, key)
        except Exception as e:
            skipped.append((key, f"load error: {e}"))
            continue

        if df[cfg.COL_STAGE].isna().all():
            skipped.append((key, "no stage labels present"))
            continue

        stages_present = set(df[cfg.COL_STAGE].dropna().unique())
        if cfg.STAGE_RELAXED not in stages_present or cfg.STAGE_STRESSED not in stages_present:
            skipped.append((key, f"missing Normal/Stress stage (has {stages_present})"))
            continue

        data[key] = df

    if verbose:
        print(f"Loaded {len(data)} usable volunteers.")
        print(f"Skipped {len(skipped)} volunteers:")
        for k, reason in skipped:
            print(f"  {k}: {reason}")

    return data, skipped