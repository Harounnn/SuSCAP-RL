import os
import json
import pandas as pd
from datetime import datetime
from collections import defaultdict

# CONFIG

GOOGLE_JSON_PATH = "data/raw/google_cluster/instance_usage/instance_usage-000000000000.json"  
CARBON_CSV_PATH  = "data/raw/carbon_intensity_2024.csv"

OUTPUT_DIR  = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "merged_timeseries.csv")

TIME_FREQ = "1min"          
CPU_TO_KWH = 0.5            
TARGET_ZONE_ID = None       

# STEP 1: Load carbon intensity
def load_carbon_intensity(csv_path):
    df = pd.read_csv(csv_path)

    df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"], utc=True)

    df = df.rename(columns={
        "Datetime (UTC)": "timestamp",
        "Carbon intensity gCO₂eq/kWh (direct)": "carbon_intensity"
    })

    if TARGET_ZONE_ID is not None:
        df = df[df["Zone id"] == TARGET_ZONE_ID]

    numeric_cols = ["carbon_intensity"]
    meta_cols = ["Zone id", "Country"]

    df_numeric = (
        df[["timestamp"] + numeric_cols]
        .set_index("timestamp")
        .resample(TIME_FREQ)
        .mean()
        .interpolate()
    )

    zone_id = df["Zone id"].iloc[0]
    country = df["Country"].iloc[0]

    df_numeric["Zone id"] = zone_id
    df_numeric["Country"] = country

    return df_numeric



# STEP 2: Stream Google JSON & aggregate usage
def stream_google_usage(json_path):
    cpu_sum = defaultdict(float)
    mem_sum = defaultdict(float)
    count   = defaultdict(int)

    base_ts = None
    target_start = pd.Timestamp("2024-01-01", tz="UTC")

    print("[INFO] Streaming Google Cluster JSON...")

    with open(json_path, "r") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)

            try:
                ts = pd.to_datetime(int(obj["start_time"]), unit="us", utc=True)

                if base_ts is None:
                    base_ts = ts

                # 🔁 REBASE TIMESTAMP
                ts = target_start + (ts - base_ts)
                ts = ts.floor(TIME_FREQ)

                avg = obj.get("average_usage", {})
                cpu = avg.get("cpus", None)
                mem = avg.get("memory", None)

                if cpu is None or mem is None:
                    continue

                cpu_sum[ts] += cpu
                mem_sum[ts] += mem
                count[ts]   += 1

            except Exception:
                continue

            if i % 5_000_000 == 0 and i > 0:
                print(f"[INFO] Processed {i:,} lines")

    rows = []
    for ts in cpu_sum:
        rows.append({
            "timestamp": ts,
            "cpu_mean": cpu_sum[ts] / count[ts],
            "mem_mean": mem_sum[ts] / count[ts]
        })

    return pd.DataFrame(rows).set_index("timestamp").sort_index()



# STEP 3: Merge & compute energy / CO₂
def merge_and_compute(cluster_df, carbon_df):
    df = cluster_df.join(carbon_df, how="inner")

    df["energy_kwh"] = df["cpu_mean"] * CPU_TO_KWH
    df["co2_emissions_g"] = df["energy_kwh"] * df["carbon_intensity"]

    return df.reset_index()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    carbon_df  = load_carbon_intensity(CARBON_CSV_PATH)
    cluster_df = stream_google_usage(GOOGLE_JSON_PATH)

    print("[INFO] Merging datasets...")
    merged = merge_and_compute(cluster_df, carbon_df)

    print("[INFO] Writing output:", OUTPUT_FILE)
    merged.to_csv(OUTPUT_FILE, index=False)

    print("[DONE] Preprocessing complete")
    print(merged.head())


if __name__ == "__main__":
    main()
