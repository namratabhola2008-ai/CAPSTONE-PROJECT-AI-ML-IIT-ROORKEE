import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.shipment_prep import FEATURES

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("data/modelling_table.csv")
X, y = df[FEATURES], df["delivery_hours"]

# 1. split 840 / 180 / 180
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

# 2. scaling without leakage
scaler = StandardScaler().fit(X_train)
tr_s, te_s = scaler.transform(X_train), scaler.transform(X_test)
pd.DataFrame({"metric": ["train_rows", "val_rows", "test_rows", "train_scaled_max_abs_mean", "test_scaled_max_abs_mean"],
              "value": [len(X_train), len(X_val), len(X_test),
                        np.abs(tr_s.mean(axis=0)).max(), np.abs(te_s.mean(axis=0)).max()]}
             ).round(4).to_csv("outputs/part_b_checks.csv", index=False)


def metrics(name, split, y_true, pred):
    mse = mean_squared_error(y_true, pred)
    return {"model": name, "split": split, "MAE": mean_absolute_error(y_true, pred), "MSE": mse,
            "RMSE": np.sqrt(mse), "R2": r2_score(y_true, pred)}


rows = []
# 3. baseline
rows.append(metrics("mean_baseline", "val", y_val, np.full(len(y_val), y_train.mean())))

# 4. tree-depth diagnostic
depth_rows = []
for d in [1, 2, 3, 4, 6, 8, 12, None]:
    t = DecisionTreeRegressor(max_depth=d, random_state=42).fit(X_train, y_train)
    depth_rows.append({"max_depth": "unlimited" if d is None else d,
                       "_d": 10**6 if d is None else d,
                       "train_rmse": np.sqrt(mean_squared_error(y_train, t.predict(X_train))),
                       "val_rmse": np.sqrt(mean_squared_error(y_val, t.predict(X_val)))})
depth = pd.DataFrame(depth_rows)
best = depth.sort_values(["val_rmse", "_d"]).iloc[0]
best_depth = None if best["max_depth"] == "unlimited" else int(best["max_depth"])
depth.drop(columns="_d").round(4).to_csv("outputs/part_b_tree_depth.csv", index=False)

# 5. four candidates
models = {
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}
for name, m in models.items():
    m.fit(X_train, y_train)
    rows.append(metrics(name, "val", y_val, m.predict(X_val)))

# 6. learning-rate experiment
lr_rows = []
for lr in [0.01, 0.1, 1.0]:
    g = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42).fit(X_train, y_train)
    p = g.predict(X_val)
    lr_rows.append({"learning_rate": lr, "val_rmse": np.sqrt(mean_squared_error(y_val, p)), "val_r2": r2_score(y_val, p)})
pd.DataFrame(lr_rows).round(4).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

# 7. CV on the two best, shared KFold, then the final model scored once on test
cand = pd.DataFrame(rows[1:]).sort_values("RMSE")
top2 = list(cand["model"].iloc[:2])
final_name = top2[0]
X_trval, y_trval = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []
for name in top2:
    s = cross_val_score(models[name], X_trval, y_trval, cv=kf)
    cv_rows.append({"model": name, "mean_r2": s.mean(), "std_r2": s.std()})
pd.DataFrame(cv_rows).round(4).to_csv("outputs/part_b_cv.csv", index=False)
rows.append(metrics(final_name, "test", y_test, models[final_name].predict(X_test)))
pd.DataFrame(rows).round(4).to_csv("outputs/part_b_metrics.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))
