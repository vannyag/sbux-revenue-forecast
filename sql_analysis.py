import sqlite3
import pandas as pd

hist = pd.read_csv("outputs/history.csv")
hist.columns = ["period", "revenue_bn"]

con = sqlite3.connect("outputs/sbux.db")
hist.to_sql("revenue", con, if_exists="replace", index=False)

query = """
WITH g AS (
  SELECT period, revenue_bn,
         LAG(revenue_bn, 4) OVER (ORDER BY period) AS rev_year_ago,
         LAG(revenue_bn, 1) OVER (ORDER BY period) AS rev_prev_q
  FROM revenue
)
SELECT period,
       ROUND(revenue_bn, 2) AS revenue_bn,
       ROUND((revenue_bn / rev_year_ago - 1) * 100, 1) AS yoy_pct,
       ROUND((revenue_bn / rev_prev_q - 1) * 100, 1) AS qoq_pct
FROM g
WHERE rev_year_ago IS NOT NULL
ORDER BY period DESC
LIMIT 12;
"""
print("YEAR-OVER-YEAR AND QUARTER-OVER-QUARTER GROWTH")
print(pd.read_sql(query, con))

q2 = """
SELECT SUBSTR(period, 5, 2) AS quarter, ROUND(AVG(revenue_bn), 2) AS avg_revenue_bn
FROM revenue GROUP BY quarter ORDER BY quarter;
"""
print("\nSEASONALITY (average revenue by quarter)")
print(pd.read_sql(q2, con))