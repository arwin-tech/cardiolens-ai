# SQL data layer (DuckDB)

Data profiling, cleaning and feature engineering in SQL, run directly on the CSV with DuckDB.

| File | What it does |
|---|---|
| `00_load.sql` | Loads `data/cardiovascular_disease.csv` as a SQL view |
| `01_profile.sql` | Data-quality profile: size, class balance, missing values, impossible readings |
| `02_features.sql` | Removes impossible blood-pressure rows and builds BMI and pulse-pressure features |
| `03_patterns.sql` | Disease rate by age band and blood-pressure category (CTE + window function) |

Run from the repo root: `pip install duckdb`, then `python train_from_sql.py`.
The script runs every SQL file in order, then trains and evaluates XGBoost on the SQL-built `features` table.
This is a separate analysis; the deployed API model is unchanged.
