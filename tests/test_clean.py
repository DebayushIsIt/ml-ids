import numpy as np
import pandas as pd
import pytest

from ids.data.clean import FAMILY_COL, LABEL_COL, clean, clean_label, load_raw


def raw_frame() -> pd.DataFrame:
    """A tiny frame with the same problems as the real CSVs."""
    return pd.DataFrame(
        {
            " Destination Port": [80, 80, 22, 443, 80, 8080],
            " Flow Bytes/s": [1.0, np.inf, 3.0, np.nan, 5.0, 6.0],
            " Fwd Header Length": [10, 20, 30, 40, 50, 60],
            " Fwd Header Length.1": [10, 20, 30, 40, 50, 60],
            " Bwd PSH Flags": [0, 0, 0, 0, 0, 0],
            " Label": [
                "BENIGN",
                "BENIGN",
                "SSH-Patator",
                "BENIGN",
                "Web Attack \x96 Brute Force",
                "Web Attack – XSS",
            ],
        }
    )


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Web Attack \x96 Brute Force", "Web Attack - Brute Force"),
        ("Web Attack � XSS", "Web Attack - XSS"),
        ("Web Attack \xe2\x80\x93 Sql Injection", "Web Attack - Sql Injection"),
        ("FTP-Patator", "FTP-Patator"),
        (" BENIGN ", "BENIGN"),
    ],
)
def test_clean_label(raw, expected):
    assert clean_label(raw) == expected


def test_clean_removes_bad_rows_and_columns():
    df, summary = clean(raw_frame())

    assert "Fwd Header Length.1" not in df.columns
    assert "Bwd PSH Flags" not in df.columns
    assert summary["rows_with_inf_or_nan"] == 2
    assert np.isfinite(df.drop(columns=[LABEL_COL, FAMILY_COL]).to_numpy(float)).all()
    assert set(df[LABEL_COL]) == {
        "BENIGN", "SSH-Patator", "Web Attack - Brute Force", "Web Attack - XSS"
    }
    assert (df[FAMILY_COL] != "Unknown").all()


def test_clean_drops_exact_duplicates():
    raw = pd.concat([raw_frame(), raw_frame()], ignore_index=True)
    df, summary = clean(raw)
    assert summary["duplicate_rows"] == 4
    assert not df.duplicated().any()


def test_benign_downsampling_keeps_all_attacks():
    raw = raw_frame()
    benign = pd.DataFrame(
        {c: raw[c].iloc[[0]].repeat(100).to_numpy() for c in raw.columns}
    )
    benign[" Destination Port"] = range(1000, 1100)  # make rows unique
    raw = pd.concat([raw, benign], ignore_index=True)

    df, _ = clean(raw, benign_frac=0.2)

    assert (df[LABEL_COL] == "BENIGN").sum() == round(101 * 0.2)
    assert (df[LABEL_COL] != "BENIGN").sum() == 3


def test_load_raw_strips_columns_and_reads_cp1252(tmp_path):
    raw = raw_frame()
    raw[" Label"] = raw[" Label"].str.replace("\x96", "\u2013")  # real files are cp1252
    raw.to_csv(tmp_path / "day.csv", index=False, encoding="cp1252")
    df = load_raw(tmp_path)
    assert "Label" in df.columns and "Flow Bytes/s" in df.columns
    assert df["Flow Bytes/s"].dtype == np.float32
    df, _ = clean(df)
    assert "Web Attack - Brute Force" in set(df[LABEL_COL])


def test_load_raw_without_files_explains(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        load_raw(tmp_path)
