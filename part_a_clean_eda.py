import os
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features, FEATURES

os.makedirs("outputs", exist_ok=True)
raw = pd.read_csv("data/raw_shipments.csv", encoding="utf-8")
hubs = pd.read_csv("data/hubs.csv", encoding="utf-8")

# "before" measurements: after removing duplicates, before any filling
dedup = raw.drop_duplicates()
sentinel_rows = int((dedup["parcel_weight_kg"] > 100).sum())
before = {
    "missing_distance_km_before": int(dedup["distance_km"].isna().sum()),
    "missing_parcel_weight_before": int(dedup["parcel_weight_kg"].isna().sum()),
    "sentinel_weight_rows": sentinel_rows,
    "missing_declared_value_before": int(dedup["declared_value"].isna().sum()),
    "missing_payment_mode_before": int(dedup["payment_mode"].isna().sum()),
    "service_type_spellings_before": int(dedup["service_type"].nunique()),
}

clean, medians = clean_shipments(raw)
spell_after = int(clean["service_type"].nunique())
df = build_features(clean, hubs)
df["is_delayed"] = (df["delivery_hours"] > df["promised_hours"]).astype(int)
df.to_csv("data/modelling_table.csv", index=False)

q1, q3 = df["delivery_hours"].quantile(0.25), df["delivery_hours"].quantile(0.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
n_out = int(((df["delivery_hours"] < lo) | (df["delivery_hours"] > hi)).sum())

checks = {
    "raw_rows": len(raw),
    "duplicate_rows_removed": len(raw) - len(dedup),
    "rows_after_dedup": len(dedup),
    **before,
    "service_type_spellings_after": spell_after,
    "median_distance_km": medians["distance_km"],
    "median_parcel_weight_kg": medians["parcel_weight_kg"],
    "median_declared_value_inr": medians["declared_value_inr"],
    "rows_after_hub_merge": len(df),
    "nulls_in_features_and_target": int(df[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum()),
    "delayed_count": int(df["is_delayed"].sum()),
    "delay_rate": df["is_delayed"].mean(),
    "hours_q1": q1, "hours_q3": q3,
    "hours_lower_bound": lo, "hours_upper_bound": hi,
    "hours_outlier_count": n_out,
    "declared_value_mean_inr": df["declared_value_inr"].mean(),
    "declared_value_median_inr": df["declared_value_inr"].median(),
    "log_declared_value_mean": df["log_declared_value"].mean(),
    "log_declared_value_median": df["log_declared_value"].median(),
}
out = pd.DataFrame({"metric": list(checks), "value": list(checks.values())})
out["value"] = out["value"].astype(float).round(4)
out.to_csv("outputs/part_a_checks.csv", index=False)

# delay evidence by group
df["distance_band"] = pd.cut(df["distance_km"], bins=[0, 200, 400, 600, 2000],
                             labels=["0-200", "201-400", "401-600", "601+"], right=True, include_lowest=True)
rows = []
for dim in ["service_type", "hub_tier", "distance_band"]:
    g = df.groupby(dim, observed=True).agg(shipments=("is_delayed", "size"), delayed=("is_delayed", "sum"),
                                           delay_rate=("is_delayed", "mean"), mean_hours=("delivery_hours", "mean"))
    for grp, r in g.iterrows():
        rows.append({"dimension": dim, "group": str(grp), "shipments": int(r["shipments"]),
                     "delayed": int(r["delayed"]), "delay_rate": r["delay_rate"], "mean_hours": r["mean_hours"]})
pd.DataFrame(rows).round(4).to_csv("outputs/part_a_delay_by_group.csv", index=False)

# correlations of numeric columns (modelling table, without the helper band)
num = df.drop(columns=["distance_band"]).select_dtypes(include="number")
num.corr().round(4).to_csv("outputs/part_a_correlations.csv")
print(out.to_string(index=False))
