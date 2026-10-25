"""Phase 4 (Task 2a/2b): build the SQLite database, load all casualty rows, execute the three
task statements and integrity tests, and save execution evidence (text + PNG)."""
import sqlite3
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import CLEAN_CSV, GUIDE_XLSX, PROCESSED_DIR, SQL_DIR, SQL_EVIDENCE_DIR, road_user_group

DB_PATH = PROCESSED_DIR / "casualty_2024.sqlite"

LOOKUPS = {  # table: (guide field, code column, label column)
    "severity": ("casualty_severity", "severity_code", "severity_label"),
    "enhanced_severity": ("enhanced_casualty_severity", "enhanced_code", "enhanced_label"),
    "casualty_class": ("casualty_class", "class_code", "class_label"),
    "sex": ("sex_of_casualty", "sex_code", "sex_label"),
    "imd_decile": ("casualty_imd_decile", "imd_code", "imd_label"),
    "distance_band": ("casualty_distance_banding", "distance_code", "distance_label"),
    "car_passenger_type": ("car_passenger", "car_passenger_code", "car_passenger_label"),
    "bus_passenger_type": ("bus_or_coach_passenger", "bus_passenger_code", "bus_passenger_label"),
    "pedestrian_location": ("pedestrian_location", "ped_location_code", "ped_location_label"),
    "pedestrian_movement": ("pedestrian_movement", "ped_movement_code", "ped_movement_label"),
    "road_worker_status": ("pedestrian_road_maintenance_worker", "road_worker_code", "road_worker_label"),
}
AGE_BANDS = [(1, "0 - 5", 0, 5), (2, "6 - 10", 6, 10), (3, "11 - 15", 11, 15), (4, "16 - 20", 16, 20),
             (5, "21 - 25", 21, 25), (6, "26 - 35", 26, 35), (7, "36 - 45", 36, 45),
             (8, "46 - 55", 46, 55), (9, "56 - 65", 56, 65), (10, "66 - 75", 66, 75),
             (11, "Over 75", 76, 120)]


def guide_codes(guide, field):
    g = guide[guide["field name"] == field].copy()
    g["code"] = pd.to_numeric(g["code/format"], errors="coerce")
    g = g.dropna(subset=["code"])
    g = g[g["code"] != -1]  # -1 is stored as NULL, not as a category
    return [(int(c), str(lbl).strip()) for c, lbl in zip(g["code"], g["label"])]


def nullable(series):
    return [None if pd.isna(v) else int(v) for v in series]


def build(con):
    con.executescript((SQL_DIR / "01_schema.sql").read_text())
    con.execute("BEGIN")
    guide = pd.read_excel(GUIDE_XLSX)
    guide = guide[guide["table"] == "casualty"]
    for table, (field, code_col, label_col) in LOOKUPS.items():
        con.executemany(f"INSERT INTO {table} ({code_col}, {label_col}) VALUES (?, ?)",
                        guide_codes(guide, field))
    con.executemany("INSERT INTO age_band VALUES (?, ?, ?, ?)", AGE_BANDS)
    con.executemany("INSERT INTO casualty_type VALUES (?, ?, ?)",
                    [(c, l, road_user_group(c)) for c, l in guide_codes(guide, "casualty_type")])

    df = pd.read_csv(CLEAN_CSV, dtype={"collision_index": str, "collision_ref_no": str}, low_memory=False)
    col = df[["collision_index", "collision_year", "collision_ref_no"]].drop_duplicates()
    con.executemany("INSERT INTO collision VALUES (?, ?, ?)", col.itertuples(index=False, name=None))

    cas = list(zip(df["collision_index"], df["casualty_reference"].astype(int),
                   df["vehicle_reference"].astype(int), df["casualty_class"].astype(int),
                   nullable(df["sex_of_casualty"]), nullable(df["age_of_casualty"]),
                   df["casualty_severity"].astype(int), nullable(df["enhanced_casualty_severity"]),
                   df["casualty_injury_based"].astype(int),
                   df["casualty_adjusted_severity_serious"].astype(float),
                   df["casualty_adjusted_severity_slight"].astype(float),
                   nullable(df["casualty_type"]), nullable(df["car_passenger"]),
                   nullable(df["bus_or_coach_passenger"]), nullable(df["casualty_imd_decile"]),
                   nullable(df["casualty_distance_banding"])))
    con.executemany("INSERT INTO casualty VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", cas)

    ped = df[df["casualty_class"] == 3]
    con.executemany("INSERT INTO pedestrian_detail VALUES (?,?,?,?,?)",
                    list(zip(ped["collision_index"], ped["casualty_reference"].astype(int),
                             nullable(ped["pedestrian_location"]), nullable(ped["pedestrian_movement"]),
                             nullable(ped["pedestrian_road_maintenance_worker"]))))
    con.execute("COMMIT")
    return df


def statements(sql_text):
    buf, out = "", []
    for line in sql_text.splitlines(keepends=True):
        if line.strip().startswith("--") and not buf.strip():
            continue
        buf += line
        if sqlite3.complete_statement(buf):
            out.append(buf.strip())
            buf = ""
    return out


def run_queries(con):
    log = []
    results = {}
    for i, stmt in enumerate(statements((SQL_DIR / "02_task_queries.sql").read_text())):
        cur = con.execute(stmt)
        first = " ".join(stmt.split())[:90]
        if cur.description:
            cols = [d[0] for d in cur.description]
            res = pd.DataFrame(cur.fetchall(), columns=cols)
            results[i] = res
            log.append(f"\n>>> {first} ...\n{res.to_string(index=False)}\n({len(res)} row(s))")
        else:
            log.append(f"\n>>> {first} ...\nOK, rows affected: {cur.rowcount}")
    return log, results


def integrity_tests(con):
    tests = {
        "invalid severity code (4)":
            "INSERT INTO casualty (collision_index, casualty_reference, vehicle_reference, casualty_class_code, "
            "severity_code, injury_based, adjusted_serious_prob, adjusted_slight_prob) "
            "VALUES ('2024TEST000001', 2, 1, 1, 4, 0, 0, 1)",
        "unknown collision (FK)":
            "INSERT INTO casualty (collision_index, casualty_reference, vehicle_reference, casualty_class_code, "
            "severity_code, injury_based, adjusted_serious_prob, adjusted_slight_prob) "
            "VALUES ('NOSUCHCOLLISION', 1, 1, 1, 3, 0, 0, 1)",
        "duplicate primary key":
            "INSERT INTO casualty (collision_index, casualty_reference, vehicle_reference, casualty_class_code, "
            "severity_code, injury_based, adjusted_serious_prob, adjusted_slight_prob) "
            "VALUES ('2024TEST000001', 1, 1, 3, 2, 0, 1, 0)",
        "pedestrian detail on a driver":
            "INSERT INTO pedestrian_detail SELECT collision_index, casualty_reference, 1, 1, 0 "
            "FROM casualty WHERE casualty_class_code = 1 LIMIT 1",
        "age out of range (150)":
            "INSERT INTO casualty (collision_index, casualty_reference, vehicle_reference, casualty_class_code, "
            "age_of_casualty, severity_code, injury_based, adjusted_serious_prob, adjusted_slight_prob) "
            "VALUES ('2024TEST000001', 3, 1, 1, 150, 3, 0, 0, 1)",
    }
    out = []
    for name, sql in tests.items():
        try:
            con.execute(sql)
            con.rollback()
            out.append(f"[UNEXPECTED SUCCESS] {name}")
        except sqlite3.DatabaseError as e:
            con.rollback()
            out.append(f"[REJECTED as expected] {name}: {type(e).__name__}: {e}")
    return out


def render_png(text, path, title):
    lines = text.splitlines()
    fig = plt.figure(figsize=(11, 0.17 * len(lines) + 0.6))
    fig.text(0.01, 0.99, title, fontsize=9, weight="bold", va="top", family="monospace")
    fig.text(0.01, 0.99 - 0.35 / (0.17 * len(lines) + 0.6), text, fontsize=7, va="top",
             family="monospace")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = sqlite3.connect(DB_PATH, isolation_level=None)  # autocommit; scripts manage transactions
    con.execute("PRAGMA foreign_keys = ON")
    df = build(con)

    counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ["collision", "casualty", "pedestrian_detail", "severity", "casualty_type"]}
    fk = con.execute("PRAGMA foreign_key_check").fetchall()

    log, results = run_queries(con)
    # Repeatability: run the script a second time; it must give identical results.
    _, results2 = run_queries(con)
    repeatable = all(results[k].equals(results2[k]) for k in results)
    tests = integrity_tests(con)

    plans = []
    for label, sql in [("(b) filter", "SELECT COUNT(*) FROM casualty c JOIN pedestrian_detail pd "
                                      "ON pd.collision_index = c.collision_index AND pd.casualty_reference = "
                                      "c.casualty_reference WHERE c.casualty_class_code = 3 AND c.severity_code "
                                      "IN (1,2) AND c.age_of_casualty >= 70 AND pd.ped_location_code IN (4,5)"),
                       ("(c) aggregate", "SELECT casualty_type_code, severity_code, COUNT(*) FROM casualty "
                                         "GROUP BY casualty_type_code, severity_code")]:
        plan = con.execute("EXPLAIN QUERY PLAN " + sql).fetchall()
        plans.append(f"{label}: " + " | ".join(r[-1] for r in plan))

    # Cross-check (c) against pandas on the cleaned CSV (+1 for the inserted test casualty).
    agg = results[max(results)]
    expected = len(df) + 1
    cross = {"sql_total": int(agg["total"].sum()), "expected_total_incl_test_row": expected,
             "sql_fatal": int(agg["fatal"].sum()), "pandas_fatal": int((df["casualty_severity"] == 1).sum()),
             "sql_serious": int(agg["serious"].sum()),
             "pandas_serious_plus_test": int((df["casualty_severity"] == 2).sum()) + 1}

    header = [f"Database: SQLite {sqlite3.sqlite_version} ({DB_PATH.name})",
              f"Loaded rows: {counts}", f"PRAGMA foreign_key_check violations: {len(fk)}"]
    text = "\n".join(header + log + ["", "Integrity tests:"] + tests +
                     ["", "Query plans:"] + plans +
                     ["", f"Repeatable on second run: {repeatable}", f"Cross-check: {cross}"])
    (SQL_EVIDENCE_DIR / "sql_execution_log.txt").write_text(text)
    for k, res in results.items():
        res.to_csv(SQL_EVIDENCE_DIR / f"query_result_{k}.csv", index=False)

    # Compact PNG evidence for statements (a), (b), (c).
    keys = sorted(results)
    render_png("\n".join([header[0], "", "(a) INSERT committed; verification SELECT:",
                          results[keys[0]].to_string(index=False), "",
                          "Integrity tests:"] + [textwrap.shorten(t, 140) for t in tests]),
               SQL_EVIDENCE_DIR / "evidence_a_insert.png", "Task 2b (a) INSERT - execution output")
    render_png("\n".join(["(b) Serious/fatal pedestrians aged 70+, crossing at locations 4/5 (first 10):",
                          results[keys[1]].to_string(index=False), "",
                          results[keys[2]].to_string(index=False), "", plans[0]]),
               SQL_EVIDENCE_DIR / "evidence_b_filter.png", "Task 2b (b) FILTER - execution output")
    render_png("\n".join(["(c) Casualties by type and severity:", agg.to_string(index=False), "",
                          f"Cross-check: {cross}", plans[1]]),
               SQL_EVIDENCE_DIR / "evidence_c_count.png", "Task 2b (c) COUNT - execution output")
    con.close()
    print(text)
    assert repeatable and not fk and cross["sql_total"] == expected
    assert all(t.startswith("[REJECTED") for t in tests)


if __name__ == "__main__":
    main()
