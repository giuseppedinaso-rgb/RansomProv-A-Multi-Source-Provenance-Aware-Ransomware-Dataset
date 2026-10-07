#!/usr/bin/env python3
"""Sweep degli iperparametri per il compito 2 (prevedibilita' del bersaglio, tabulare).

Stesse feature e stessa divisione di tabular.py. Selezione senza toccare il test:
  * il periodo di training (< 2025-07-01) e' diviso in sotto-training
    (< 2025-01-01) e validazione (primo semestre 2025);
  * Optuna (TPE, seed 0), N_TRIALS prove per modello, obiettivo = log-loss di
    validazione;
  * la configurazione migliore si riaddestra su tutto il training e si valuta
    sul test (etichette osservate).
Modelli: regressione logistica (C), LightGBM, XGBoost, CatBoost.
Uso: python tabular_sweep.py [n_trials]   -> benchmark/results/tabular_sweep_summary.csv
"""
import sys, json, warnings
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import log_loss
import tabular as T

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
VAL_START = pd.Timestamp("2025-01-01")
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 30


def data(version, target):
    obs, classes = T.load("observed", target)
    ev, _ = T.load(version, target, [c for c in classes if c != "other"] if target == "country" else None)
    X = T.features(ev, target, classes).set_index("event_id")
    ev = ev.set_index("event_id")
    lab, true = ev[target], obs.set_index("event_id")[target]
    d = ev.d.reindex(X.index)
    tr = X.index[(d < pd.Timestamp(T.CUTOFF)).values & lab.reindex(X.index).notna().values]
    te = X.index[(d >= pd.Timestamp(T.CUTOFF)).values & true.reindex(X.index).notna().values]
    sub = tr[d.loc[tr] < VAL_START]; va = tr[d.loc[tr] >= VAL_START]
    cat = ["gang", "gang_last"]
    Xc = X.copy()
    for c in cat:
        Xc[c] = X[c].astype(pd.CategoricalDtype(sorted(set(X[c]))))
    return X, Xc, lab, true, classes, tr, te, sub, va, cat


def make(name, p, cat, num, classes, seed=0):
    if name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import make_pipeline
        pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), cat), ("num", StandardScaler(), num)])
        return make_pipeline(pre, LogisticRegression(max_iter=3000, C=p["C"]))
    if name == "LightGBM":
        import lightgbm as lgb
        return lgb.LGBMClassifier(verbose=-1, random_state=seed, n_jobs=4, **p)
    if name == "XGBoost":
        import xgboost as xgb
        return xgb.XGBClassifier(enable_categorical=True, tree_method="hist", random_state=seed, n_jobs=4, **p)
    from catboost import CatBoostClassifier
    return CatBoostClassifier(verbose=0, random_seed=seed, cat_features=cat, thread_count=4, **p)


def space(t, name):
    if name == "LogisticRegression":
        return {"C": t.suggest_float("C", 1e-3, 10, log=True)}
    if name == "LightGBM":
        return {"n_estimators": t.suggest_int("n_estimators", 100, 800), "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": t.suggest_int("num_leaves", 8, 128, log=True), "min_child_samples": t.suggest_int("min_child_samples", 10, 200, log=True),
                "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True), "subsample": t.suggest_float("subsample", 0.5, 1.0),
                "subsample_freq": 1, "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0)}
    if name == "XGBoost":
        return {"n_estimators": t.suggest_int("n_estimators", 100, 800), "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": t.suggest_int("max_depth", 2, 10), "min_child_weight": t.suggest_float("min_child_weight", 1, 50, log=True),
                "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True), "subsample": t.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0)}
    return {"iterations": t.suggest_int("iterations", 200, 1000), "learning_rate": t.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "depth": t.suggest_int("depth", 3, 8), "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1, 30, log=True)}


def fit_proba(name, p, X, Xc, lab, idx_fit, idx_pred, classes, cat, num):
    m = make(name, p, cat, num, classes)
    D = X if name in ("LogisticRegression", "CatBoost") else Xc
    y = lab.loc[idx_fit].values
    if name == "XGBoost":
        enc = {c: i for i, c in enumerate(classes)}
        m.fit(D.loc[idx_fit], np.array([enc[v] for v in y]))
        return T.align(m.predict_proba(D.loc[idx_pred]), np.arange(len(classes)), np.arange(len(classes)))
    m.fit(D.loc[idx_fit], y)
    return T.align(m.predict_proba(D.loc[idx_pred]), m.classes_, classes)


def main():
    rows = []
    for target in ("sector", "country"):
        for version in ("enriched", "observed"):
            X, Xc, lab, true, classes, tr, te, sub, va, cat = data(version, target)
            num = [c for c in X.columns if c not in cat]
            for name in ("LogisticRegression", "LightGBM", "XGBoost", "CatBoost"):
                st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
                def obj(t):
                    P = fit_proba(name, space(t, name), X, Xc, lab, sub, va, classes, cat, num)
                    return log_loss(lab.loc[va].values, np.clip(P, 1e-9, 1), labels=classes)
                st.optimize(obj, n_trials=N_TRIALS if name != "LogisticRegression" else 10)
                P = fit_proba(name, st.best_params if name != "LightGBM" else {**st.best_params, "subsample_freq": 1},
                              X, Xc, lab, tr, te, classes, cat, num)
                r = T.evaluate(true.loc[te].values, P, classes, f"{name} (tuned)")
                r.update(version=version, target=target, best_val_logloss=st.best_value, params=json.dumps(st.best_params))
                rows.append(r); print(target, version, name, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(T.OUT / "tabular_sweep_summary.csv", index=False)
    print(R.drop(columns=["params"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
