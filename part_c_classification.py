import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix
from src.shipment_prep import FEATURES, NOISE_COLUMNS

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("data/modelling_table.csv")
X, y = df[FEATURES], df["is_delayed"]

# 1. stratified split
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
print("sizes", len(X_train), len(X_val), len(X_test), "delayed", y_train.sum(), y_val.sum(), y_test.sum())


def scores(y_true, pred):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    acc = (tn + tp) / (tn + fp + fn + tp)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# 2. baseline + three classifiers (0.5 rule)
rows = [{"model": "blind_all_on_time", **scores(y_val, np.zeros(len(y_val), dtype=int))}]
lr = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]).fit(X_train, y_train)
rows.append({"model": "logistic_regression", **scores(y_val, lr.predict(X_val))})
best_d, best_f1, best_pred = None, -1, None
for d in [2, 3, 4, 5, 6, 8, 10]:
    t = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42).fit(X_train, y_train)
    s = scores(y_val, t.predict(X_val))
    if s["f1"] > best_f1:                       # strict '>' keeps the smaller d on ties
        best_d, best_f1, best_s = d, s["f1"], s
rows.append({"model": f"decision_tree_depth_{best_d}", **best_s})
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
rows.append({"model": "random_forest_default", **scores(y_val, rf.predict(X_val))})
pd.DataFrame(rows).round(4).to_csv("outputs/part_c_models.csv", index=False)

# 3. which columns carry signal
sk = SelectKBest(f_classif, k="all").fit(X_train, y_train)
fs = pd.DataFrame({"feature": FEATURES, "f_score": sk.scores_, "rf_importance": rf.feature_importances_})
fs["f_rank"] = fs["f_score"].rank(ascending=False, method="first").astype(int)
fs["is_noise"] = fs["feature"].isin(NOISE_COLUMNS)
fs = fs[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]].sort_values("f_rank")
fs.round(4).to_csv("outputs/part_c_feature_scores.csv", index=False)
sel_rows = []
for k in [5, 7, 10, 12, 17]:
    p = Pipeline([("selector", SelectKBest(f_classif, k=k)),
                  ("model", RandomForestClassifier(n_estimators=100, random_state=42))]).fit(X_train, y_train)
    sel_rows.append({"k": k, "val_f1": scores(y_val, p.predict(X_val))["f1"]})
pd.DataFrame(sel_rows).round(4).to_csv("outputs/part_c_selectk.csv", index=False)

# 4. tuning with k = 7 (selector fitted on training rows only; test rows untouched)
sel7 = SelectKBest(f_classif, k=7).fit(X_train, y_train)
Xtr7, Xva7 = sel7.transform(X_train), sel7.transform(X_val)
grid = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}
gs = GridSearchCV(RandomForestClassifier(random_state=42), grid, cv=5).fit(Xtr7, y_train)
rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), grid, n_iter=9, cv=5, random_state=42).fit(Xtr7, y_train)
default7 = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr7, y_train)
f1_of = lambda m: scores(y_val, m.predict(Xva7))["f1"]
tune = [
    {"search": "grid_search", "candidates": 27, "cv_folds": 5, "total_fits": 135,
     "best_params": str(gs.best_params_), "best_cv_accuracy": gs.best_score_, "val_f1": f1_of(gs.best_estimator_)},
    {"search": "randomized_search", "candidates": 9, "cv_folds": 5, "total_fits": 45,
     "best_params": str(rs.best_params_), "best_cv_accuracy": rs.best_score_, "val_f1": f1_of(rs.best_estimator_)},
    {"search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0, "total_fits": 1,
     "best_params": str({"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}),
     "best_cv_accuracy": np.nan, "val_f1": f1_of(default7)},
]
tune = pd.DataFrame(tune).round(4)
tune.to_csv("outputs/part_c_tuning.csv", index=False)

# 5. final configuration: highest val_f1; ties -> default, then grid, then randomized
priority = ["default_forest_same_7_features", "grid_search", "randomized_search"]
t = tune.set_index("search").loc[priority]
winner = t["val_f1"].idxmax()                    # idxmax returns the first max in priority order
params = {"default_forest_same_7_features": {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
          "grid_search": gs.best_params_, "randomized_search": rs.best_params_}[winner]
final = Pipeline([("selector", SelectKBest(f_classif, k=7)),
                  ("model", RandomForestClassifier(random_state=42, **params))]).fit(X_train, y_train)

# 6. alert threshold on validation
pv = final.predict_proba(X_val)[:, 1]
thr_rows = []
for th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    thr_rows.append({"threshold": th, **scores(y_val, (pv >= th).astype(int))})
thr = pd.DataFrame(thr_rows)
ok = thr[thr["recall"] >= 0.80]
chosen = float(ok["threshold"].max()) if len(ok) else 0.1
thr.round(4).to_csv("outputs/part_c_thresholds.csv", index=False)

# 7. one test evaluation
pt = final.predict_proba(X_test)[:, 1]
pd.DataFrame([{"threshold": chosen, **scores(y_test, (pt >= chosen).astype(int))}]).round(4).to_csv(
    "outputs/part_c_final_test.csv", index=False)
pd.DataFrame([{"selector_k": 7, "n_estimators": params["n_estimators"],
               "max_depth": "unlimited" if params["max_depth"] is None else params["max_depth"],
               "min_samples_leaf": params["min_samples_leaf"], "threshold": chosen,
               "chosen_from": winner}]).to_csv("outputs/part_c_final_config.csv", index=False)
print(tune.to_string(index=False))
