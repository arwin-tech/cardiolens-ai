-- Data-quality profile: size, class balance, missing values, impossible readings
SELECT
  COUNT(*)                                       AS total_rows,
  ROUND(AVG(cardio), 3)                          AS disease_rate,
  COUNT(*) - COUNT(ap_hi)                        AS missing_ap_hi,
  MIN(ap_hi) AS min_ap_hi, MAX(ap_hi) AS max_ap_hi,
  MIN(ap_lo) AS min_ap_lo, MAX(ap_lo) AS max_ap_lo,
  CAST(SUM(CASE WHEN ap_hi NOT BETWEEN 60 AND 250
                 OR ap_lo NOT BETWEEN 30 AND 180
                 OR ap_lo > ap_hi THEN 1 ELSE 0 END) AS INTEGER) AS impossible_bp_rows,
  COUNT(*) - COUNT(DISTINCT id)                  AS duplicate_ids
FROM raw;
