# Big Data Analytics Summative Assessment

Code, original and processed data, and saved outputs for an analysis of the 2024 UK Department
for Transport (DfT) STATS19 casualty dataset. The Word document is maintained
separately; this repository does not build documents or submission archives.

The analysis covers severity associations (investigation A), age and road-user
groups (B), predictive modelling (C), and a normalised SQLite database with SQL
queries and integrity checks.

## 1. Set up Python

The saved outputs were generated with Python **3.14.5**. Use that version and the
pinned dependencies for the closest reproduction; other versions have not been
verified. Run the following commands from the repository root on macOS/Linux:

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

On Windows, create the environment with `py -3.14 -m venv .venv` and replace
`.venv/bin/python` in subsequent commands with `.venv\Scripts\python.exe`.
No separate database server or diagramming software is required.

## 2. Original inputs

The original casualty CSV and data-guide workbook are included in
[`data/unprocessed/`](data/unprocessed/), alongside the cleaned data and saved
outputs elsewhere in the repository. No additional data download is needed for
a full rebuild.

The input paths, relative to the repository root, are:

```text
data/unprocessed/
  dft-road-casualty-statistics-casualty-2024.csv
  dft-road-casualty-statistics-road-safety-open-dataset-data-guide-2024.xlsx
```

The workbook must contain the casualty code definitions, including the columns
`table`, `field name`, `code/format` and `label`. Input paths are defined in
[`src/config.py`](src/config.py); change them there if your files live elsewhere.
The scripts read the original files without modifying them. Keep this folder
unchanged; derived data belongs in [`data/processed/`](data/processed/) and
analysis results in [`outputs/`](outputs/).

The source is the [DfT road safety open data page](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data).
If replacing the inputs, use the same 2024 assessment files: source revisions
may differ and produce different results.

The original CSV has **128,272 rows**, **23 columns**, and **100,927 distinct
collisions**. Its SHA-256, recorded in
[`provenance.json`](outputs/tables/provenance.json), is:

```text
4a5f364306bf322491f9b4996df3891021424aee3a766283870f5f994431cc06
```

## 3. Run the pipeline

From the repository root, run the following command to execute all five steps.
The `&&` operators stop the pipeline if any step fails:

```bash
.venv/bin/python src/01_profile_data.py &&
.venv/bin/python src/02_clean_data.py &&
.venv/bin/python src/03_analyse_model.py &&
.venv/bin/python src/04_sql_demo.py &&
.venv/bin/python src/05_diagrams.py
```

| Script | Inputs | Generated outputs |
|---|---|---|
| [`01_profile_data.py`](src/01_profile_data.py) | Original CSV and guide workbook | Provenance, data-quality profile, severity frequencies and consistency checks in `outputs/tables/` |
| [`02_clean_data.py`](src/02_clean_data.py) | Original CSV | Cleaned CSV, cleaning log and before/after dimensions |
| [`03_analyse_model.py`](src/03_analyse_model.py) | Cleaned CSV | Investigation A-C tables, model metrics, `summary.json` and figures 1-7 |
| [`04_sql_demo.py`](src/04_sql_demo.py) | Cleaned CSV, guide workbook and SQL files | SQLite database, query-result CSVs, execution log and three evidence images |
| [`05_diagrams.py`](src/05_diagrams.py) | Schema/architecture encoded in the script | ERD (figure 8) and architecture diagram (figure 9) |

Output directories are created automatically. Rerunning scripts overwrites their
generated files; the SQL script deletes and rebuilds the existing SQLite database.
Modelling includes grouped cross-validation and repeated permutation importance,
so it takes longer than the other steps and can use all available CPU cores.

### Rerun only the analysis and diagrams

If you have not changed the input data or cleaning rules, you can reuse the
included cleaned CSV and rerun only the modelling and diagrams:

```bash
.venv/bin/python src/03_analyse_model.py &&
.venv/bin/python src/05_diagrams.py
```

To rebuild the SQL database, run step 4 with the included guide workbook. The
existing database can be inspected with a SQLite client, but is ignored by Git;
step 4 is how to regenerate it.

## 4. Check the outcome

Compare regenerated files against the saved results below. The random seed is
**42**. Training/test partitions and cross-validation are grouped by collision
to avoid placing casualties from the same collision in both partitions.
Small floating-point differences can occur across platforms or library versions.

| Check | Saved result | Where to check |
|---|---|---|
| Cleaned row count | 128,272; no rows removed | [`cleaning_before_after.csv`](outputs/tables/cleaning_before_after.csv) |
| Severity counts | Fatal: 1,602; serious: 26,040; slight: 100,630 | [`summary.json`](outputs/tables/summary.json) |
| Killed or seriously injured (KSI) rate | 21.5495% | [`summary.json`](outputs/tables/summary.json) |
| Train/test rows | 102,617 / 25,655; zero shared collisions | [`summary.json`](outputs/tables/summary.json) |
| Selected model | Logistic regression | [`summary.json`](outputs/tables/summary.json) |
| Logistic regression test metrics | Balanced accuracy: 0.615676; ROC AUC: 0.660910; KSI recall: 0.592874 | [`C_test_metrics.csv`](outputs/tables/C_test_metrics.csv) |
| SQL foreign-key violations | 0 | [`sql_execution_log.txt`](outputs/sql_evidence/sql_execution_log.txt) |
| SQL repeatability and integrity | Repeatable: `True`; all five invalid inserts rejected | [`sql_execution_log.txt`](outputs/sql_evidence/sql_execution_log.txt) |
| SQL aggregate total | 128,273, including one synthetic demonstration casualty | [`query_result_11.csv`](outputs/sql_evidence/query_result_11.csv) |

The synthetic SQL casualty is not added to the cleaned CSV or modelling data.
It explains the extra SQL row and the SQL serious-casualty count of 26,041.

Cleaning converts missing-value code `-1` to null, preserves valid unknown
categories, corrects inconsistent pedestrian fields and removes the LSOA
quasi-identifier. Modelling handles missing predictors inside its preprocessing
pipeline and excludes outcome-derived severity fields from predictors. These
are retrospective associations among recorded casualties, not causal effects or
estimates of risk per journey.

## Repository layout

| Path | Contents |
|---|---|
| [`src/`](src/) | Five numbered pipeline scripts and shared configuration |
| [`requirements.txt`](requirements.txt) | Pinned Python dependencies |
| [`sql/01_schema.sql`](sql/01_schema.sql) | Normalised SQLite schema, constraints and indexes |
| [`sql/02_task_queries.sql`](sql/02_task_queries.sql) | Demonstration insert, pedestrian filter and severity counts |
| [`data/unprocessed/`](data/unprocessed/) | Original casualty CSV and data-guide workbook; read-only pipeline inputs |
| [`data/processed/`](data/processed/) | Cleaned analysis CSV and locally generated SQLite database |
| [`outputs/tables/`](outputs/tables/) | Statistics, model metrics, cleaning records and provenance |
| [`outputs/figures/`](outputs/figures/) | Nine analysis and design figures |
| [`outputs/sql_evidence/`](outputs/sql_evidence/) | SQL execution log, query results and evidence images |

The source data is attributed to the UK Department for Transport and is supplied
under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
