-- Exploration: disease rate by age band and blood-pressure category,
-- plus each category's share of its age band (CTE + window function)
WITH labelled AS (
  SELECT
    (age_years // 10) * 10                       AS age_band,
    CASE WHEN ap_hi >= 140 OR ap_lo >= 90 THEN 'high'
         WHEN ap_hi >= 120               THEN 'elevated'
         ELSE 'normal' END                       AS bp_category,
    cardio
  FROM features
)
SELECT
  age_band,
  bp_category,
  COUNT(*)                                       AS patients,
  ROUND(AVG(cardio), 3)                          AS disease_rate,
  ROUND(COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY age_band), 3)
                                                 AS share_of_age_band
FROM labelled
GROUP BY age_band, bp_category
ORDER BY age_band, bp_category;
