import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("data/modelling_table.csv")
hubs = pd.read_csv("data/hubs.csv")

# 1. hub profile (groupby sorts by ascending hub_id)
prof = df.groupby("hub_id").agg(shipments=("hub_id", "size"), mean_delivery_hours=("delivery_hours", "mean"),
                                delay_rate=("is_delayed", "mean"), mean_distance_km=("distance_km", "mean"),
                                mean_n_stops=("n_stops", "mean")).reset_index().sort_values("hub_id").reset_index(drop=True)
prof.round(4).to_csv("outputs/hub_profile.csv", index=False)

# 2. K-Means on the scaled profile
cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
Z = StandardScaler().fit_transform(prof[cols])
km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Z)
prof["cluster"] = km.labels_

# 3. separation evidence (276 pairs)
D = euclidean_distances(Z)
iu = np.triu_indices(len(Z), k=1)
sep = []
for K in [2, 3, 4, 5]:
    lab = KMeans(n_clusters=K, random_state=42, n_init=10).fit(Z).labels_
    same = (lab[:, None] == lab[None, :])[iu]
    d = D[iu]
    sizes = sorted(np.bincount(lab), reverse=True)
    sep.append({"K": K, "across_over_within": d[~same].mean() / d[same].mean(),
                "cluster_sizes": "-".join(str(s) for s in sizes)})
pd.DataFrame(sep).round(4).to_csv("outputs/part_d_separation.csv", index=False)

# 4. names by mean delay rate: lowest Standard, middle Watch, highest Escalate
order = prof.groupby("cluster")["delay_rate"].mean().sort_values().index
name_of = dict(zip(order, ["Standard", "Watch", "Escalate"]))
prof["segment"] = prof["cluster"].map(name_of)
seg = prof[["hub_id", "segment"]].merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")
seg.to_csv("outputs/hub_segments.csv", index=False)
sp = prof.groupby("segment").agg(hubs=("hub_id", "size"), mean_delivery_hours=("mean_delivery_hours", "mean"),
                                 delay_rate=("delay_rate", "mean"), mean_distance_km=("mean_distance_km", "mean"),
                                 mean_n_stops=("mean_n_stops", "mean")).reindex(["Standard", "Watch", "Escalate"]).reset_index()
sp.round(4).to_csv("outputs/segment_profile.csv", index=False)
implied = {1: "Standard", 2: "Watch", 3: "Escalate"}
seg[seg["segment"] != seg["hub_tier"].map(implied)].to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)
print(sp.round(4).to_string(index=False))
