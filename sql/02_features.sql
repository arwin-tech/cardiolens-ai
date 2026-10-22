-- Clean the data and engineer features; the model trains on this table
CREATE OR REPLACE TABLE features AS
SELECT
  id,
  CAST(ROUND(age / 365.25) AS INTEGER)           AS age_years,
  gender, height, weight, ap_hi, ap_lo,
  cholesterol, gluc, smoke, alco, active,
  ROUND(weight / POWER(height / 100.0, 2), 1)    AS bmi,
  ap_hi - ap_lo                                  AS pulse_pressure,
  cardio
FROM raw
WHERE ap_hi BETWEEN 60 AND 250
  AND ap_lo BETWEEN 30 AND 180
  AND ap_lo <= ap_hi;
