#!/usr/bin/env python3
"""Benchmark 2 — prevedibilita' del bersaglio (classificazione tabulare).

Domanda: al momento in cui una gang pubblica una rivendicazione, conoscendo solo
il contesto passato, quanto si puo' prevedere il SETTORE e il PAESE della vittima?

Etichette:
  sector   13 classi;
  country  9 paesi piu' frequenti nel training + 'other' (10 classi, compatibile
           con TabPFN).
Feature, tutte calcolate solo con eventi STRETTAMENTE precedenti:
  gang (categorica), anno, mese, giorno della settimana; eta' della gang (giorni
  dalla prima rivendicazione), numero di rivendicazioni precedenti, giorni
  dall'ultima; distribuzione del bersaglio nelle ultime 180 rivendicazioni
  della gang e nelle ultime 500 globali; bersaglio dell'ultima vittima della gang.
Nessuna informazione sulla vittima stessa (nome, dominio, fonti).

Divisione temporale: training fino al 30/06/2025, test dal 01/07/2025.
Due versioni: le etichette (e le feature storiche) vengono da events_observed o
da events_enriched; il TEST usa sempre e solo etichette osservate, cosi' le due
versioni sono confrontate sulla stessa verita'.

Modelli: classe piu' frequente; prior della gang (classe piu' frequente nel
passato della gang); regressione logistica; LightGBM; XGBoost; CatBoost;
TabPFN (solo paese: max 10 classi; training sottocampionato a 10.000 righe).
Metriche: accuratezza, macro-F1, log-loss, top-3.
"""
import os, sys, warnings
from pathlib import Path
from collections import defaultdict, deque, Counter
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, top_k_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark" / "results"
CUTOFF = "2025-07-01"


def features(ev, target, classes):
    ev = ev.sort_values(["d", "event_id"]).reset_index(drop=True)
    K = len(classes); idx = {c: i for i, c in enumerate(classes)}
    gang_hist, glob_hist = defaultdict(lambda: deque(maxlen=180)), deque(maxlen=500)
    first, count, last_d, last_lab = {}, Counter(), {}, {}
    rows = []
    for day, block in ev.groupby("d", sort=True):
        feats = []
        for r in block.itertuples():                       # prima calcola, poi aggiorna: niente stesso giorno
            g = r.gang
            gh = np.zeros(K); gl = np.zeros(K)
            for lab in gang_hist[g]: gh[lab] += 1
            for lab in glob_hist: gl[lab] += 1
            feats.append({"event_id": r.event_id, "gang": g, "year": day.year, "month": day.month, "dow": day.dayofweek,
                          "gang_age": (day - first[g]).days if g in first else -1, "gang_n": count[g],
                          "gang_gap": (day - last_d[g]).days if g in last_d else -1,
                          "gang_last": last_lab.get(g, "none"),
                          **{f"g_{i}": v / max(gh.sum(), 1) for i, v in enumerate(gh)},
                          **{f"G_{i}": v / max(gl.sum(), 1) for i, v in enumerate(gl)},
                          "gang_hist_n": gh.sum()})
        rows += feats
        for r in block.itertuples():
            g = r.gang; lab = getattr(r, target)
            first.setdefault(g, day); count[g] += 1; last_d[g] = day
            if lab in idx:
                gang_hist[g].append(idx[lab]); glob_hist.append(idx[lab]); last_lab[g] = lab
    return pd.DataFrame(rows)


def load(version, target, classes_ref=None):
    ev = pd.read_json(ROOT / "release" / f"events_{version}.json")
    ev["d"] = pd.to_datetime(ev.discovered)
    ev = ev[ev.d >= "2021-01-01"]
    if target == "country":
        top = classes_ref or list(ev[ev.d < CUTOFF].country.value_counts().head(9).index)
        ev["country"] = ev.country.where(ev.country.isin(top) | ev.country.isna(), "other")
        classes = sorted(top + ["other"])          # scikit-learn richiede classi ordinate
    else:
        classes = sorted(ev.sector.dropna().unique())
    return ev, classes


def evaluate(y, proba, classes, name):
    pred = np.array(classes)[proba.argmax(1)]
    return {"model": name, "accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
            "log_loss": log_loss(y, np.clip(proba, 1e-9, 1), labels=classes),
            "top3": top_k_accuracy_score(y, proba, k=3, labels=classes)}


def run(version, target, use_tabpfn=os.environ.get("NO_TABPFN") is None):
    obs, classes = load("observed", target)
    ev, _ = load(version, target, [c for c in classes if c != "other"] if target == "country" else None)
    X = features(ev, target, classes)
    ev = ev.set_index("event_id"); Xi = X.set_index("event_id")
    lab_train = ev[target]; lab_true = obs.set_index("event_id")[target]
    train_ids = [e for e in Xi.index if ev.at[e, "d"] < pd.Timestamp(CUTOFF) and pd.notna(lab_train.get(e))]
    test_ids = [e for e in Xi.index if ev.at[e, "d"] >= pd.Timestamp(CUTOFF) and pd.notna(lab_true.get(e))]
    Xtr, Xte = Xi.loc[train_ids], Xi.loc[test_ids]
    ytr, yte = lab_train.loc[train_ids].values, lab_true.loc[test_ids].values
    cat = ["gang", "gang_last"]; num = [c for c in Xtr.columns if c not in cat]
    res = []
    # baseline: classe piu' frequente
    p = pd.Series(ytr).value_counts(normalize=True).reindex(classes, fill_value=0).values
    res.append(evaluate(yte, np.tile(p, (len(yte), 1)), classes, "Majority/prior"))
    # baseline: prior della gang
    gp = Xte[[f"g_{i}" for i in range(len(classes))]].values
    gp = np.where(Xte.gang_hist_n.values[:, None] > 0, gp, p)
    res.append(evaluate(yte, 0.9 * gp + 0.1 * p, classes, "Gang prior"))
    # regressione logistica
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), cat),
                             ("num", StandardScaler(), num)])
    lr = make_pipeline(pre, LogisticRegression(max_iter=2000, C=1.0))
    lr.fit(Xtr, ytr)
    res.append(evaluate(yte, align(lr.predict_proba(Xte), lr.classes_, classes), classes, "LogisticRegression"))
    # alberi
    Xtr_c, Xte_c = Xtr.copy(), Xte.copy()
    for c in cat:
        cats = pd.CategoricalDtype(sorted(set(Xtr[c]) | set(Xte[c])))
        Xtr_c[c] = Xtr[c].astype(cats); Xte_c[c] = Xte[c].astype(cats)
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=0)
    m.fit(Xtr_c, ytr)
    res.append(evaluate(yte, align(m.predict_proba(Xte_c), m.classes_, classes), classes, "LightGBM"))
    import xgboost as xgb
    lab = {c: i for i, c in enumerate(classes)}
    m = xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6, enable_categorical=True,
                          tree_method="hist", random_state=0)
    m.fit(Xtr_c, np.array([lab[v] for v in ytr]))
    res.append(evaluate(yte, align(m.predict_proba(Xte_c), np.arange(len(classes)), np.arange(len(classes))), classes, "XGBoost"))
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=600, learning_rate=0.08, depth=6, verbose=0, random_seed=0, cat_features=cat)
    m.fit(Xtr, ytr)
    res.append(evaluate(yte, align(m.predict_proba(Xte), m.classes_, classes), classes, "CatBoost"))
    if use_tabpfn and len(classes) <= 10:
        from tabpfn import TabPFNClassifier
        sub = Xtr.sample(min(10000, len(Xtr)), random_state=0)
        enc = {c: {v: i for i, v in enumerate(sorted(set(Xtr[c]) | set(Xte[c])))} for c in cat}
        f = lambda D: D.assign(**{c: D[c].map(enc[c]) for c in cat}).values.astype(float)
        m = TabPFNClassifier(device="cuda", random_state=0)
        m.fit(f(sub), lab_train.loc[sub.index].values)
        proba = np.vstack([m.predict_proba(f(Xte.iloc[i:i + 2000])) for i in range(0, len(Xte), 2000)])
        res.append(evaluate(yte, align(proba, m.classes_, classes), classes, "TabPFN (10k)"))
    out = pd.DataFrame(res).assign(version=version, target=target, n_train=len(ytr), n_test=len(yte))
    return out


def align(proba, have, want):
    have = list(have); out = np.zeros((proba.shape[0], len(want)))
    for j, c in enumerate(want):
        if c in have:
            out[:, j] = proba[:, have.index(c)]
    s = out.sum(1, keepdims=True); s[s == 0] = 1
    return out / s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    allres = []
    for target in ("sector", "country"):
        for version in ("observed", "enriched"):
            r = run(version, target)
            print(r.round(3).to_string(index=False), flush=True)
            allres.append(r)
    R = pd.concat(allres).round(4)
    R.to_csv(OUT / "tabular_summary.csv", index=False)


if __name__ == "__main__":
    main()
