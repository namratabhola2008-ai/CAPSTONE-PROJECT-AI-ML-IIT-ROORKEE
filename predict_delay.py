import joblib
import pandas as pd
from src.shipment_prep import clean_shipments, build_features

new = pd.read_csv("data/new_shipments.csv", encoding="utf-8")
hubs = pd.read_csv("data/hubs.csv", encoding="utf-8")
model = joblib.load("models/delay_model.pkl")
threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
imp = pd.read_csv("models/impute_values.csv")
medians = {r["column"]: float(r["median"]) for _, r in imp.iterrows()}
segments = pd.read_csv("outputs/hub_segments.csv")[["hub_id", "segment"]]

clean, _ = clean_shipments(new, medians)
feat = build_features(clean, hubs)
feat["delay_probability"] = model.predict_proba(feat[order])[:, 1].round(4)
feat["flagged"] = (feat["delay_probability"] >= threshold).astype(int)
feat = feat.merge(segments, on="hub_id", how="left")
feat["action"] = "No action"
feat.loc[feat["flagged"] == 1, "action"] = "Proactive customer call"
feat.loc[(feat["flagged"] == 1) & (feat["segment"] == "Escalate"), "action"] = "Escalate to hub manager"
feat[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]].to_csv(
    "outputs/new_shipment_actions.csv", index=False)
print(feat["action"].value_counts().to_string())
