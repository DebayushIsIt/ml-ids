"""Turn the cleaned CICIDS2017 table into model-ready splits.

Usage:
    python -m ids.features.prepare --top-k 25

Steps, in order:
    1. Drop families with too few rows to learn from (Infiltration, Heartbleed).
    2. Drop features that leak the label (Destination Port).
    3. Drop one of each pair of near-duplicate features (|correlation| > 0.98).
    4. Rank the rest by Random Forest importance on a stratified sample and
       keep the top k. The list goes to configs/features.yaml, which the
       models, the API and the live sensor all read.
    5. Stratified train / validation / test split (70 / 15 / 15) on Family,
       saved to parquet.

The prediction target is Family (Benign, DoS, DDoS, PortScan, Brute force,
Web attack, Bot). Label is kept alongside it for per-attack reporting.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ids.data.clean import FAMILY_COL, LABEL_COL, SEED

log = logging.getLogger(__name__)

RARE_FAMILIES = ["Infiltration", "Heartbleed"]
# Destination Port identifies the attacked service, so the model would learn
# "port 21 = FTP-Patator" instead of anything about the traffic itself.
LEAKY_FEATURES = ["Destination Port"]
CORR_THRESHOLD = 0.98
FEATURES_YAML = Path("configs/features.yaml")


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in (LABEL_COL, FAMILY_COL)]


def drop_rare_and_leaky(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df[FAMILY_COL].isin(RARE_FAMILIES)]
    return df.drop(columns=[c for c in LEAKY_FEATURES if c in df.columns])


def correlated_features(
    df: pd.DataFrame, features: list[str], threshold: float = CORR_THRESHOLD
) -> list[str]:
    """Features to drop so that no remaining pair has |corr| > threshold.

    Walks the columns in order and drops a column if it is too correlated
    with any column already kept, so the earlier (usually simpler) one stays.
    """
    corr = df[features].astype(np.float64).corr().abs().fillna(0.0)
    kept: list[str] = []
    dropped: list[str] = []
    for col in features:
        if any(corr.loc[col, k] > threshold for k in kept):
            dropped.append(col)
        else:
            kept.append(col)
    return dropped


def rank_features(
    df: pd.DataFrame,
    features: list[str],
    sample_size: int = 200_000,
    seed: int = SEED,
) -> pd.Series:
    """Random Forest importance of each feature, highest first."""
    if len(df) > sample_size:
        df, _ = train_test_split(
            df, train_size=sample_size, stratify=df[FAMILY_COL], random_state=seed
        )
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    rf.fit(df[features], df[FAMILY_COL])
    return pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)


def split(
    df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15, seed: int = SEED
) -> dict[str, pd.DataFrame]:
    """Stratified train / val / test split on Family."""
    train, rest = train_test_split(
        df, test_size=val_size + test_size, stratify=df[FAMILY_COL], random_state=seed
    )
    val, test = train_test_split(
        rest,
        test_size=test_size / (val_size + test_size),
        stratify=rest[FAMILY_COL],
        random_state=seed,
    )
    return {"train": train, "val": val, "test": test}


def save_feature_config(
    features: list[str],
    importance: pd.Series,
    dropped_correlated: list[str],
    path: Path = FEATURES_YAML,
) -> None:
    config = {
        "label": LABEL_COL,
        "family": FAMILY_COL,
        "target": FAMILY_COL,
        "dropped_families": RARE_FAMILIES,
        "drop_leaky": LEAKY_FEATURES,
        "dropped_correlated": dropped_correlated,
        "features": features,
        "importance": {f: round(float(importance[f]), 5) for f in features},
    }
    header = (
        "# Written by `python -m ids.features.prepare`. Do not edit by hand.\n"
        "# The model, the API schema and the live sensor all read `features`.\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + yaml.safe_dump(config, sort_keys=False))


def load_feature_list(path: str | Path = FEATURES_YAML) -> list[str]:
    features = yaml.safe_load(Path(path).read_text())["features"]
    if not features:
        raise ValueError(f"{path} has no features yet. Run `python -m ids.features.prepare`.")
    return features


def prepare(
    df: pd.DataFrame, top_k: int = 25, seed: int = SEED
) -> tuple[dict[str, pd.DataFrame], list[str], pd.Series, list[str]]:
    """Steps 1 to 5. Feature selection only looks at the training split."""
    df = drop_rare_and_leaky(df)
    splits = split(df, seed=seed)

    candidates = feature_columns(df)
    dropped_corr = correlated_features(splits["train"], candidates)
    candidates = [c for c in candidates if c not in dropped_corr]

    importance = rank_features(splits["train"], candidates, seed=seed)
    features = list(importance.index[:top_k])

    cols = features + [LABEL_COL, FAMILY_COL]
    splits = {name: part[cols].reset_index(drop=True) for name, part in splits.items()}
    return splits, features, importance, dropped_corr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clean", default="data/processed/cicids2017_clean.parquet")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = pd.read_parquet(args.clean)
    splits, features, importance, dropped_corr = prepare(df, top_k=args.top_k, seed=args.seed)

    out = Path(args.out)
    for name, part in splits.items():
        part.to_parquet(out / f"{name}.parquet", index=False)
        log.info("%-5s %8d rows", name, len(part))
    save_feature_config(features, importance, dropped_corr)

    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    importance.rename("importance").to_csv(reports / "feature_importance.csv")
    counts = {
        name: part[FAMILY_COL].value_counts().to_dict() for name, part in splits.items()
    }
    (reports / "split_counts.json").write_text(json.dumps(counts, indent=2))

    log.info("dropped as near-duplicates: %s", dropped_corr)
    log.info("kept %d features:\n  %s", len(features), "\n  ".join(features))


if __name__ == "__main__":
    main()
