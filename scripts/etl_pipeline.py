from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# ✅ Create SparkSession
spark = SparkSession.builder \
    .appName("Ecommerce ETL Pipeline") \
    .getOrCreate()

# ✅ Load main transactions data
df_txn = spark.read.option("header", True).option("inferSchema", True).csv("data/ecommerce_transactions.csv")
print("✅ Transaction Data Schema:")
df_txn.printSchema()
df_txn.show(5)

# ✅ Load country-region mapping
df_region = spark.read.option("header", True).option("inferSchema", True).csv("data/country_region.csv")
print("✅ Country-Region Mapping:")
df_region.show(5)

# -------------------------------------
# ✅ Step 3: Clean the Transaction Data
# -------------------------------------

# 1. Drop duplicates
df_txn_clean = df_txn.dropDuplicates()

# 2. Drop null customer_id
df_txn_clean = df_txn_clean.filter(col("customer_id").isNotNull())

# 3. Filter quantity > 0
df_txn_clean = df_txn_clean.filter(col("quantity") > 0)

# ✅ Show cleaned data
print("✅ Cleaned Data:")
df_txn_clean.show(5)

from pyspark.sql.functions import to_date, year, month, dayofmonth

# ✅ Enrich: Add order_value column
df_enriched = df_txn_clean.withColumn("order_value", col("quantity") * col("unit_price"))

# ✅ Convert order_date to DateType if not already
df_enriched = df_enriched.withColumn("order_date", to_date(col("order_date")))

# ✅ Extract year, month, day
df_enriched = df_enriched \
    .withColumn("order_year", year(col("order_date"))) \
    .withColumn("order_month", month(col("order_date"))) \
    .withColumn("order_day", dayofmonth(col("order_date")))

# ✅ Show enriched data
print("✅ Enriched Data:")
df_enriched.select("order_date", "order_year", "order_month", "order_day", "order_value").show(5)

from pyspark.sql.functions import sum as _sum

# ✅ Step 5: Country Revenue Summary
df_country_sales = df_enriched.groupBy("country") \
    .agg(_sum("order_value").alias("total_revenue")) \
    .orderBy("total_revenue", ascending=False)

print("💰 Total Revenue by Country:")
df_country_sales.show(10, truncate=False)

from pyspark.sql.window import Window
from pyspark.sql.functions import rank

# 1. Group and sum order_value per customer per country
df_customer_country = df_enriched.groupBy("country", "customer_id") \
    .agg(_sum("order_value").alias("total_spent"))

# 2. Create window spec: partition by country, order by total_spent descending
window_spec = Window.partitionBy("country").orderBy(col("total_spent").desc())

# 3. Rank customers
df_ranked = df_customer_country.withColumn("rank", rank().over(window_spec))

# 4. Filter only top-ranked (i.e., rank == 1)
df_top_customers = df_ranked.filter(col("rank") == 1).drop("rank")

print("👑 Top Customers Per Country:")
df_top_customers.show(10, truncate=False)
# ✅ Join country-region mapping
df_joined = df_enriched.join(df_region, on="country", how="left")

# ✅ Aggregate total revenue by region
df_region_sales = df_joined.groupBy("region") \
    .agg(_sum("order_value").alias("total_revenue")) \
    .orderBy("total_revenue", ascending=False)

print("🌎 Total Revenue by Region:")
df_region_sales.show(truncate=False)
# ------------------------------------------
# ✅ Step 8: Monthly Category Pivot using SQL
# ------------------------------------------

# Register enriched DF as temp view
df_enriched.createOrReplaceTempView("transactions")

# Step 1: SQL for total revenue per month & category
df_monthly_category_pivot = spark.sql("""
    SELECT
        order_month,
        category,
        ROUND(SUM(quantity * unit_price), 2) AS total_revenue
    FROM transactions
    GROUP BY order_month, category
""")

# Step 2: Pivot category into columns
df_pivot = df_monthly_category_pivot.groupBy("order_month").pivot("category").sum("total_revenue")

# Step 3: Show pivot table
print("📊 Monthly Revenue by Category (Pivoted):")
df_pivot.orderBy("order_month").show()
# ----------------------------------
# ✅ Step 9: Price Band Count (SQL)
# ----------------------------------

df_price_band_counts = spark.sql("""
    SELECT 
        CASE 
            WHEN order_value < 100 THEN '0-100'
            WHEN order_value < 500 THEN '100-500'
            WHEN order_value < 1000 THEN '500-1000'
            ELSE '1000+'
        END AS price_band,
        COUNT(*) AS order_count
    FROM transactions
    GROUP BY price_band
    ORDER BY price_band
""")

print("💸 Order Count by Price Band:")
df_price_band_counts.show()
# ----------------------------------------
# ✅ Step 10: Partition Tuning Check
# ----------------------------------------

import time
from pyspark.sql.functions import sum as _sum

# 1️⃣ Check original partitions
print("🧾 Default partition count:", df_enriched.rdd.getNumPartitions())

# 2️⃣ Default aggregation timing
start_default = time.time()
df_enriched.groupBy("country").agg(_sum("order_value").alias("total_revenue")).collect()
end_default = time.time()
print(f"⏱️ Default Partition Time: {end_default - start_default:.2f} seconds")

# 3️⃣ Repartitioned DataFrame by country
df_repartitioned = df_enriched.repartition("country")
print("🔄 Repartitioned partition count:", df_repartitioned.rdd.getNumPartitions())

start_custom = time.time()
df_repartitioned.groupBy("country").agg(_sum("order_value").alias("total_revenue")).collect()
end_custom = time.time()
print(f"⚙️ Custom Partition Time: {end_custom - start_custom:.2f} seconds")

# 4️⃣ Coalesce example before writing (optional)
df_coalesced = df_enriched.coalesce(2)
print("🔽 Coalesced partition count:", df_coalesced.rdd.getNumPartitions())

# ----------------------------------------
# ✅ Step 11: Cache vs Recompute
# ----------------------------------------

import time
from pyspark.sql.functions import sum as _sum

# ❌ Recompute every time — no cache
start_recompute = time.time()
for _ in range(3):
    df_enriched.groupBy("category").agg(_sum("order_value").alias("total_revenue")).collect()
end_recompute = time.time()
print(f"🌀 Time without cache: {end_recompute - start_recompute:.2f} seconds")

# ✅ Cache DataFrame in memory
df_enriched.cache()

# Run same aggregation multiple times
start_cache = time.time()
for _ in range(3):
    df_enriched.groupBy("category").agg(_sum("order_value").alias("total_revenue")).collect()
end_cache = time.time()
print(f"⚡ Time with cache: {end_cache - start_cache:.2f} seconds")

# 🧹 Optional: Remove cached data
df_enriched.unpersist()

# ----------------------------------------
# ✅ Step 12: Write Gold Output – Parquet
# ----------------------------------------

# Create output directory (if not already exists)
import os
os.makedirs("output", exist_ok=True)

# Save Country Revenue Summary
df_country_sales.write.mode("overwrite") \
    .option("compression", "snappy") \
    .parquet("output/country_revenue_summary")

# Save Region Revenue Summary
df_region_sales.write.mode("overwrite") \
    .option("compression", "snappy") \
    .parquet("output/region_revenue_summary")

# Save Top Customers per Country
df_top_customers.write.mode("overwrite") \
    .option("compression", "snappy") \
    .parquet("output/top_customers_per_country")

print("✅ Gold data successfully written as Parquet files with Snappy compression!")