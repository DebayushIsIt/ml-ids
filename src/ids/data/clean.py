"""Load and clean the CICIDS2017 MachineLearningCVE CSVs.

Usage:
    python -m ids.data.clean --raw data/raw/MachineLearningCVE --out data/processed

Steps, in order:
    1. Read every CSV, strip whitespace from column names, downcast numbers.
    2. Clean labels (the Web Attack rows carry a broken en dash).
    3. Drop columns that duplicate another column (e.g. "Fwd Header Length.1").
    4. Replace +/-inf with NaN, then drop rows with any NaN.
    5. Drop exact duplicate rows.
    6. Drop constant columns.
    7. Optionally downsample Benign traffic.
    8. Add a coarse attack "Family" column and save to parquet.

A summary of how many rows/columns each step removed is written next to the
parquet file, so the README numbers are reproducible.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

LABEL_COL = "Label"
FAMILY_COL = "Family"
SEED = 42

# Day 2 groups labels into these families. Infiltration and Heartbleed have
# too few rows to learn from and are dropped there, not here.
LABEL_FAMILIES = {
    "BENIGN": "Benign",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "FTP-Patator": "Brute force",
    "SSH-Patator": "Brute force",
    "Web Attack - Brute Force": "Web attack",
    "Web Attack - XSS": "Web attack",
    "Web Attack - Sql Injection": "Web attack",
    "Bot": "Bot",
    "Infiltration": "Infiltration",
    "Heartbleed": "Heartbleed",
}


def clean_label(label: str) -> str:
    """Normalise one raw label.

    The Web Attack labels contain an en dash that shows up as "\\x96",
    "\\u2013", "\\ufffd" or mojibake like "â\\x80\\x93" depending on how the
    file was decoded. Any run of non-ASCII characters becomes a plain "-".
    """
    label = re.sub(r"[^\x20-\x7E]+", "-", str(label))
    label = re.sub(r"\s+", " ", label).strip()
    return label


def read_csv(path: Path) -> pd.DataFrame:
    """Read one CICIDS2017 CSV with tidy column names and compact dtypes."""
    # latin-1 never fails to decode; clean_label fixes the en dash afterwards.
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        if col == LABEL_COL:
            continue
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype(np.float32)
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def load_raw(raw_dir: str | Path) -> pd.DataFrame:
    """Load every CSV under raw_dir into one DataFrame."""
    paths = sorted(Path(raw_dir).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No CSV files in {raw_dir}. See data/README.md for the download steps."
        )
    frames = []
    for path in paths:
        df = read_csv(path)
        log.info("loaded %-60s %9d rows", path.name, len(df))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def duplicate_columns(df: pd.DataFrame) -> list[str]:
    """Return columns whose values are identical to an earlier column."""
    seen: dict[int, list[str]] = {}
    dupes = []
    for col in df.columns:
        key = int(pd.util.hash_pandas_object(df[col], index=False).sum())
        for other in seen.get(key, []):
            if df[col].equals(df[other]):
                dupes.append(col)
                break
        else:
            seen.setdefault(key, []).append(col)
    return dupes


def constant_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != LABEL_COL and df[c].nunique(dropna=False) <= 1]


def clean(
    df: pd.DataFrame, benign_frac: float = 1.0, seed: int = SEED
) -> tuple[pd.DataFrame, dict]:
    """Run cleaning steps 2 to 8 on a raw DataFrame.

    Returns the cleaned frame and a summary dict of what each step removed.
    """
    summary: dict = {"raw_rows": len(df), "raw_cols": df.shape[1]}
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Rows with no label are blank padding at the end of some CSVs.
    before = len(df)
    df = df[df[LABEL_COL].notna()]
    df[LABEL_COL] = df[LABEL_COL].map(clean_label)
    summary["rows_without_label"] = before - len(df)

    dupes = duplicate_columns(df.drop(columns=LABEL_COL))
    df = df.drop(columns=dupes)
    summary["duplicate_columns"] = dupes

    features = df.columns.drop(LABEL_COL)
    # Column by column, so we never hold a float64 copy of the whole table.
    bad_rows = np.zeros(len(df), dtype=bool)
    n_inf = n_nan = 0
    for col in features:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        is_inf, is_nan = np.isinf(values), np.isnan(values)
        n_inf += int(is_inf.sum())
        n_nan += int(is_nan.sum())
        bad_rows |= is_inf | is_nan
    df = df[~bad_rows]
    summary["inf_values"] = n_inf
    summary["nan_values"] = n_nan
    summary["rows_with_inf_or_nan"] = int(bad_rows.sum())

    before = len(df)
    df = df.drop_duplicates()
    summary["duplicate_rows"] = before - len(df)

    consts = constant_columns(df)
    df = df.drop(columns=consts)
    summary["constant_columns"] = consts

    if benign_frac < 1.0:
        benign = df[LABEL_COL] == "BENIGN"
        kept = df[benign].sample(frac=benign_frac, random_state=seed)
        summary["benign_rows_dropped_by_sampling"] = int(benign.sum()) - len(kept)
        df = pd.concat([kept, df[~benign]]).sort_index()
    summary["benign_frac"] = benign_frac

    unknown = sorted(set(df[LABEL_COL]) - set(LABEL_FAMILIES))
    if unknown:
        log.warning("labels with no family mapping: %s", unknown)
    df[FAMILY_COL] = df[LABEL_COL].map(LABEL_FAMILIES).fillna("Unknown")

    df = df.reset_index(drop=True)
    summary["clean_rows"] = len(df)
    summary["clean_feature_cols"] = df.shape[1] - 2
    summary["label_counts"] = df[LABEL_COL].value_counts().to_dict()
    return df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", default="data/raw/MachineLearningCVE")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument(
        "--benign-frac", type=float, default=0.2,
        help="fraction of Benign rows to keep (1.0 keeps all)",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df, summary = clean(load_raw(args.raw), benign_frac=args.benign_frac, seed=args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "cicids2017_clean.parquet", index=False)
    (out / "cleaning_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("saved %d rows x %d cols to %s", *df.shape, out / "cicids2017_clean.parquet")
    log.info(json.dumps({k: v for k, v in summary.items() if k != "label_counts"}, indent=2))


if __name__ == "__main__":
    main()
