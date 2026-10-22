-- Load the raw CSV as a SQL view (DuckDB reads the file directly, no server needed)
CREATE OR REPLACE VIEW raw AS
SELECT * FROM read_csv('data/cardiovascular_disease.csv', delim = ';', header = true);
