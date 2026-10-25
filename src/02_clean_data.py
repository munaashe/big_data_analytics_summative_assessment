"""Phase 2: scripted, auditable cleaning of the casualty file.

* Raw file is read-only; the cleaned file is written to data/processed.
* -1 ("Data missing or out of range" in the guide) is converted to missing. Code 9
  ("unknown (self reported)") is a documented valid category and is preserved.
* Primary outcome: police-recorded casualty_severity (1 Fatal, 2 Serious, 3 Slight).
  enhanced_casualty_severity and the adjusted-severity probabilities are derived from the
  outcome and are retained only for sensitivity checks, never as predictors.
* lsoa_of_casualty is dropped from the analysis file (high-cardinality quasi-identifier).
* No rows are removed: (collision_index, casualty_reference) is unique and no full
  duplicates exist. Rows with missing predictors are kept and handled in modelling.
"""
import numpy as np
import pandas as pd

from config import (AGE_BAND, CASUALTY_CLASS, CASUALTY_TYPE, CLEAN_CSV, DISTANCE, IMD,
                    PED_LOCATION, PED_MOVEMENT, RAW_CSV, SEVERITY, SEX, TAB_DIR, road_user_group)

SENTINEL_FIELDS = ["sex_of_casualty", "age_of_casualty", "age_band_of_casualty",
                   "pedestrian_location", "pedestrian_movement", "car_passenger",
                   "bus_or_coach_passenger", "pedestrian_road_maintenance_worker",
                   "casualty_type", "casualty_imd_decile", "enhanced_casualty_severity",
                   "casualty_distance_banding"]


def main():
    raw = pd.read_csv(RAW_CSV, dtype={"collision_index": str, "collision_ref_no": str,
                                      "lsoa_of_casualty": str})
    df = raw.copy()
    log = []

    def record(step, field, n, rule):
        log.append({"step": step, "field": field, "rows_affected": int(n), "rule": rule})

    assert not df.duplicated(["collision_index", "casualty_reference"]).any()
    record("key check", "collision_index+casualty_reference", 0,
           "Composite key unique; no rows removed")

    for col in SENTINEL_FIELDS:
        n = int((df[col] == -1).sum())
        df[col] = df[col].replace(-1, np.nan).astype("Int64")
        record("sentinel to missing", col, n, "-1 = 'Data missing or out of range' (guide)")

    bad_ped = (df["casualty_class"] != 3) & (df["pedestrian_location"].fillna(0) > 0)
    record("consistency", "pedestrian_location/movement", bad_ped.sum(),
           "Pedestrian fields set to missing for non-pedestrian casualties with pedestrian codes")
    df.loc[bad_ped, ["pedestrian_location", "pedestrian_movement"]] = pd.NA

    record("drop field", "lsoa_of_casualty", len(df),
           "High-cardinality small-area code; not needed for analysis; privacy minimisation")
    df = df.drop(columns=["lsoa_of_casualty"])

    df["severity_label"] = df["casualty_severity"].map(SEVERITY)
    df["ksi"] = (df["casualty_severity"] <= 2).astype(int)
    df["class_label"] = df["casualty_class"].map(CASUALTY_CLASS)
    df["sex_label"] = df["sex_of_casualty"].map(SEX).fillna("Missing")
    df["age_band_label"] = df["age_band_of_casualty"].map(AGE_BAND).fillna("Missing")
    df["casualty_type_label"] = df["casualty_type"].map(CASUALTY_TYPE).fillna("Missing")
    df["road_user_group"] = df["casualty_type"].fillna(-1).astype(int).map(road_user_group)
    df["imd_label"] = df["casualty_imd_decile"].map(IMD).fillna("Missing")
    df["distance_label"] = df["casualty_distance_banding"].map(DISTANCE).fillna("Missing")
    df["ped_location_label"] = df["pedestrian_location"].map(PED_LOCATION).fillna("Missing")
    df["ped_movement_label"] = df["pedestrian_movement"].map(PED_MOVEMENT).fillna("Missing")

    # Collision-context features derived only from the supplied casualty file.
    df["casualties_in_collision"] = df.groupby("collision_index")["casualty_reference"] \
        .transform("size")
    df["max_vehicle_ref_in_collision"] = df.groupby("collision_index")["vehicle_reference"] \
        .transform("max")
    record("derive", "casualties_in_collision", len(df),
           "Count of casualty records sharing collision_index")
    record("derive", "max_vehicle_ref_in_collision", len(df),
           "Highest vehicle_reference among casualties in collision (lower bound on vehicles)")

    df.to_csv(CLEAN_CSV, index=False)
    pd.DataFrame(log).to_csv(TAB_DIR / "cleaning_log.csv", index=False)

    before_after = pd.DataFrame({
        "raw_rows": [len(raw)], "clean_rows": [len(df)],
        "raw_columns": [raw.shape[1]], "clean_columns": [df.shape[1]],
    })
    before_after.to_csv(TAB_DIR / "cleaning_before_after.csv", index=False)
    print(pd.DataFrame(log).to_string(index=False))
    print(before_after.to_string(index=False))


if __name__ == "__main__":
    main()
