import numpy as np
import pandas as pd

from ids.data.clean import FAMILY_COL, LABEL_COL
from ids.features.prepare import (
    correlated_features,
    load_feature_list,
    prepare,
    save_feature_config,
)


def clean_frame(n_per_family: int = 200, seed: int = 0) -> pd.DataFrame:
    """Cleaned-looking data where `signal` separates the families."""
    rng = np.random.default_rng(seed)
    families = ["Benign", "DoS", "PortScan", "Bot"]
    rows = []
    for i, fam in enumerate(families):
        n = n_per_family
        rows.append(
            pd.DataFrame(
                {
                    "Destination Port": np.full(n, 80 + i),
                    "signal": rng.normal(i * 10, 1, n),
                    "noise_a": rng.normal(0, 1, n),
                    "noise_b": rng.normal(0, 1, n),
                    LABEL_COL: fam.upper(),
                    FAMILY_COL: fam,
                }
            )
        )
    df = pd.concat(rows, ignore_index=True)
    df["signal_copy"] = df["signal"] * 2 + 1
    rare = df.iloc[:3].assign(**{FAMILY_COL: "Heartbleed", LABEL_COL: "Heartbleed"})
    return pd.concat([df, rare], ignore_index=True)


def test_correlated_features_keeps_first_of_pair():
    df = clean_frame()
    feats = ["signal", "noise_a", "signal_copy", "noise_b"]
    assert correlated_features(df, feats) == ["signal_copy"]


def test_prepare_drops_leaky_rare_and_correlated():
    splits, features, importance, dropped = prepare(clean_frame(), top_k=2)

    assert "Destination Port" not in features
    assert dropped == ["signal_copy"]
    assert features[0] == "signal"
    assert len(features) == 2
    for part in splits.values():
        assert list(part.columns) == features + [LABEL_COL, FAMILY_COL]
        assert "Heartbleed" not in set(part[FAMILY_COL])


def test_split_is_stratified_70_15_15():
    splits, *_ = prepare(clean_frame(n_per_family=400), top_k=3)
    sizes = {k: len(v) for k, v in splits.items()}
    assert sum(sizes.values()) == 1600
    assert abs(sizes["train"] / 1600 - 0.70) < 0.01
    assert abs(sizes["val"] - sizes["test"]) <= 1
    for part in splits.values():
        shares = part[FAMILY_COL].value_counts(normalize=True)
        assert np.allclose(shares, 0.25, atol=0.01)


def test_feature_config_round_trip(tmp_path):
    _, features, importance, dropped = prepare(clean_frame(), top_k=3)
    path = tmp_path / "features.yaml"
    save_feature_config(features, importance, dropped, path)
    assert load_feature_list(path) == features
