"""Run the SQL pipeline in DuckDB, then train and evaluate a model on its output.
Run from the repo root:  pip install duckdb   then   python train_from_sql.py"""
from pathlib import Path
import duckdb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

con = duckdb.connect()
for sql_file in sorted(Path("sql").glob("*.sql")):
    print(f"\n-- {sql_file.name}")
    body = "\n".join(l for l in sql_file.read_text().splitlines() if not l.strip().startswith("--")).strip()
    if body.upper().startswith(("SELECT", "WITH")):   # queries: show their rows
        print(con.sql(body).df().to_string(index=False))
    else:                                             # setup statements: just run them
        con.execute(body)

df = con.sql("SELECT * FROM features").df()
X, y = df.drop(columns=["id", "cardio"]), df["cardio"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = XGBClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
print(f"\nRows kept after SQL cleaning: {len(df):,}")
print(f"Test ROC-AUC on SQL-built features: {auc:.3f}")
