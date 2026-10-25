"""Phase 1: profile the supplied casualty file (read-only) and write a data-quality report."""
import hashlib
import json

import pandas as pd

from config import GUIDE_XLSX, RAW_CSV, TAB_DIR


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    df = pd.read_csv(RAW_CSV, dtype={"collision_index": str, "collision_ref_no": str,
                                     "lsoa_of_casualty": str})
    guide = pd.read_excel(GUIDE_XLSX)
    guide = guide[guide["table"] == "casualty"]

    # Map dataset column names to guide field names where they differ slightly.
    alias = {"casualty_adjusted_severity_serious": "casualty_adjusted_serious",
             "casualty_adjusted_severity_slight": "casualty_adjusted_slight"}

    provenance = {
        "file": RAW_CSV.name,
        "source": "UK Department for Transport, Road Safety Data (STATS19), casualty table 2024",
        "licence": "Open Government Licence v3.0",
        "sha256": sha256(RAW_CSV),
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate_rows_full": int(df.duplicated().sum()),
        "duplicate_keys_collision_casualty": int(
            df.duplicated(["collision_index", "casualty_reference"]).sum()),
        "distinct_collisions": int(df["collision_index"].nunique()),
        "collision_years": sorted(df["collision_year"].unique().tolist()),
    }
    (TAB_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2))

    rows = []
    for col in df.columns:
        s = df[col]
        g = guide[guide["field name"] == alias.get(col, col)]
        codes = pd.to_numeric(g["code/format"], errors="coerce").dropna().astype(int).tolist()
        n_minus1 = int((s.astype(str) == "-1").sum())
        invalid = None
        if len(codes) > 1 and pd.api.types.is_integer_dtype(s):
            invalid = int((~s.isin(codes)).sum())
        rows.append({
            "field": col, "dtype": str(s.dtype), "n_null": int(s.isna().sum()),
            "n_minus1_sentinel": n_minus1, "pct_minus1": round(100 * n_minus1 / len(s), 2),
            "n_distinct": int(s.nunique()), "guide_codes": len(codes) if codes else "free",
            "n_codes_not_in_guide": invalid,
            "example_values": ", ".join(map(str, s.value_counts().index[:5])),
        })
    dq = pd.DataFrame(rows)
    dq.to_csv(TAB_DIR / "data_quality_profile.csv", index=False)

    sev = df["casualty_severity"].value_counts().sort_index()
    sev.to_frame("count").assign(pct=lambda x: (100 * x["count"] / x["count"].sum()).round(2)) \
        .to_csv(TAB_DIR / "severity_class_frequency.csv")

    # Age consistency: age vs age band
    bands = [(1, 0, 5), (2, 6, 10), (3, 11, 15), (4, 16, 20), (5, 21, 25), (6, 26, 35), (7, 36, 45),
             (8, 46, 55), (9, 56, 65), (10, 66, 75), (11, 76, 200)]
    known = df[df["age_of_casualty"] >= 0]
    exp = pd.Series(-1, index=known.index)
    for b, lo, hi in bands:
        exp[(known["age_of_casualty"] >= lo) & (known["age_of_casualty"] <= hi)] = b
    mismatch = int((exp != known["age_band_of_casualty"]).sum())
    checks = {
        "age_min": int(known["age_of_casualty"].min()), "age_max": int(known["age_of_casualty"].max()),
        "age_band_mismatch": mismatch,
        "pedestrian_class_vs_type_mismatch": int(
            ((df["casualty_class"] == 3) != (df["casualty_type"] == 0)).sum()),
        "ped_location_nonzero_for_non_ped": int(
            ((df["casualty_class"] != 3) & (df["pedestrian_location"] > 0)).sum()),
        "enhanced_severity_missing_pct": round(
            100 * (df["enhanced_casualty_severity"] == -1).mean(), 2),
        "injury_based_pct": round(100 * df["casualty_injury_based"].mean(), 2),
    }
    (TAB_DIR / "consistency_checks.json").write_text(json.dumps(checks, indent=2))

    print(json.dumps(provenance, indent=2))
    print(json.dumps(checks, indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(dq.drop(columns="example_values").to_string(index=False))
        print(sev)


if __name__ == "__main__":
    main()
