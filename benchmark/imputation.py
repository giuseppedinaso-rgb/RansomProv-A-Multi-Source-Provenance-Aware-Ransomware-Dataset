#!/usr/bin/env python3
"""Benchmark 5 — imputazione di paese e settore mancanti.

Domanda: quanto bene si possono ricostruire paese e settore quando mancano, e
come si collocano le regole usate per events_enriched rispetto ai metodi
standard di imputazione?

Protocollo: sugli eventi con paese E settore osservati si nasconde il valore
di un attributo in un sottoinsieme di test e lo si ricostruisce.
Due schemi di mascheramento (5 ripetizioni ciascuno, 20% degli eventi):
  MCAR     casuale uniforme;
  pattern  probabilita' proporzionale al tasso di mancanza reale della cella
           (anno, combinazione di fonti) dell'evento: imita dove mancano davvero
           i valori (2021-22, eventi solo RansomLook).
Metodi (stesse feature per tutti: gang, anno, mese, ccTLD del dominio,
combinazione di fonti, e l'altro attributo):
  Mode           classe piu' frequente
  GangMode       classe piu' frequente della gang (fallback Mode)
  kNN            KNNImputer su codifica one-hot (k=15)
  MICE           IterativeImputer (BayesianRidge) su codifica ordinale, arrotondato
  MissForest     IterativeImputer con RandomForest (stile missForest)
  LightGBM       classificatore diretto sulle feature
  Rules          cascata di events_enriched senza LLM ne' sito (ccTLD ->
                 descrizione -> stessa vittima), valutata solo dove decide;
                 copertura e precisione riportate a parte.
Metriche: accuratezza e macro-F1 (copertura piena); per Rules anche copertura.
Uscita: benchmark/results/imputation.csv
"""
import os, json, warnings
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
REPS, FRAC = 5, 0.2


def cctld(d):
    if not isinstance(d, str) or "." not in d:
        return "none"
    t = d.rsplit(".", 1)[-1]
    return t if len(t) == 2 else "generic"


def prepare():
    ev = pd.read_json(ROOT / "release" / "events_observed.json")
    ev["d"] = pd.to_datetime(ev.discovered)
    ev["year"], ev["month"] = ev.d.dt.year, ev.d.dt.month
    ev["tld"] = ev.domain.map(cctld)
    ev["combo"] = ev.sources.map(lambda s: "+".join(sorted(s)))
    miss_rate = ev.assign(m=ev.country.isna() | ev.sector.isna()).groupby(["year", "combo"]).m.mean()
    full = ev[ev.country.notna() & ev.sector.notna()].copy()
    full["p_pattern"] = full.set_index(["year", "combo"]).index.map(miss_rate).fillna(0).values + 1e-3
    top = full.country.value_counts().head(30).index
    full["country30"] = full.country.where(full.country.isin(top), "other")
    keys = pd.read_json(os.environ.get("IMP_KEYS", ROOT / "build" / "event_keys.json")).set_index("event_id")
    full["desc"] = full.event_id.map(keys._desc).fillna("")
    full["vkeys"] = full.event_id.map(lambda e: list(keys._vkeys.get(e, [])) + list(keys._dkeys.get(e, [])))
    return full


def rules_country(test, train):
    """Cascata deterministica di events_enriched: ccTLD, descrizione, stessa vittima."""
    import sys
    sys.path.insert(0, str(ROOT))
    if os.environ.get("IMP_CODE"):          # codice della pipeline di una versione precedente (es. v1.0)
        sys.path.insert(0, os.environ["IMP_CODE"])
    from normalize import cctld_to_iso2, single_country_mention
    table = {}
    for ks, c in zip(train.vkeys, train.country):
        for k in ks:
            table.setdefault(k, Counter())[c] += 1
    out = []
    for d, desc, ks in zip(test.domain, test.desc, test.vkeys):
        v = cctld_to_iso2(d) or single_country_mention(desc)
        if not v:
            c = Counter()
            for k in ks:
                c.update(table.get(k, {}))
            v = c.most_common(1)[0][0] if c else None
        out.append(v)
    return pd.Series(out, index=test.index)


def rules_sector(test, train):
    table = {}
    for ks, s in zip(train.vkeys, train.sector):
        for k in ks:
            table.setdefault(k, Counter())[s] += 1
    out = []
    for ks in test.vkeys:
        c = Counter()
        for k in ks:
            c.update(table.get(k, {}))
        out.append(c.most_common(1)[0][0] if c else None)
    return pd.Series(out, index=test.index)


def run_rep(full, target, scheme, rep):
    rng = np.random.default_rng(rep)
    n = int(len(full) * FRAC)
    p = None if scheme == "MCAR" else (full.p_pattern / full.p_pattern.sum()).values
    test_idx = rng.choice(full.index.values, size=n, replace=False, p=p)
    te = full.loc[test_idx]; tr = full.drop(index=test_idx)
    other = "sector" if target == "country30" else "country30"
    feats = ["gang", "year", "month", "tld", "combo", other]
    y_te = te[target].values
    res = []
    preds = {}
    def add(m, pred):
        preds[m] = np.asarray(pred, dtype=object)
        res.append({"target": target, "scheme": scheme, "rep": rep, "method": m,
                    "accuracy": accuracy_score(y_te, pred),
                    "macro_f1": f1_score(y_te, pred, average="macro"), "coverage": 1.0})
    mode = tr[target].mode()[0]
    add("Mode", np.full(len(te), mode))
    gm = tr.groupby("gang")[target].agg(lambda s: s.mode()[0])
    add("GangMode", te.gang.map(gm).fillna(mode).values)
    # codifiche
    cat = pd.concat([tr, te])[feats + [target]].astype(str)
    codes = {c: {v: i for i, v in enumerate(sorted(cat[c].unique()))} for c in cat.columns}
    enc = lambda D: np.column_stack([D[c].astype(str).map(codes[c]).values for c in feats + [target]]).astype(float)
    Xtr, Xte = enc(tr), enc(te); Xte_m = Xte.copy(); Xte_m[:, -1] = np.nan
    inv = {i: v for v, i in codes[target].items()}; K = len(inv)
    dec = lambda col: [inv[int(min(max(round(v), 0), K - 1))] for v in col]
    # kNN su one-hot
    oh = pd.get_dummies(cat[feats], columns=feats, sparse=False).astype(float).values
    ytr_oh = pd.get_dummies(tr[target].astype(str)).reindex(columns=sorted(codes[target]), fill_value=0).astype(float).values
    A = np.hstack([oh, np.vstack([ytr_oh, np.full((len(te), ytr_oh.shape[1]), np.nan)])])
    sub = rng.choice(len(tr), size=min(len(tr), 15000), replace=False)
    rows = np.concatenate([sub, np.arange(len(tr), len(tr) + len(te))])
    imp = KNNImputer(n_neighbors=15).fit_transform(A[rows])[-len(te):, oh.shape[1]:]
    add("kNN", [sorted(codes[target])[j] for j in imp.argmax(1)])
    # MICE e MissForest su codifica ordinale
    M = np.vstack([Xtr, Xte_m])
    for name, est in (("MICE", BayesianRidge()),
                      ("MissForest", RandomForestRegressor(n_estimators=50, max_depth=12, n_jobs=4, random_state=rep))):
        out = IterativeImputer(estimator=est, max_iter=5, random_state=rep).fit_transform(M)[-len(te):, -1]
        add(name, dec(out))
    # LightGBM diretto
    Ctr, Cte = tr[feats].astype("category"), te[feats].astype("category")
    for c in feats:
        cats = pd.CategoricalDtype(sorted(set(tr[c].astype(str)) | set(te[c].astype(str))))
        Ctr[c] = tr[c].astype(str).astype(cats); Cte[c] = te[c].astype(str).astype(cats)
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, verbose=-1, random_state=rep, n_jobs=4).fit(Ctr, tr[target])
    add("LightGBM", m.predict(Cte))
    # regole di events_enriched
    rp = (rules_country if target == "country30" else rules_sector)(te, tr)
    if target == "country30":
        top = set(full.country30.unique())
        rp = rp.map(lambda c: c if c is None or c in top else "other")
    ok = rp.notna().values
    if ok.any():
        res.append({"target": target, "scheme": scheme, "rep": rep, "method": "Rules (enriched, no LLM/site)",
                    "accuracy": accuracy_score(y_te[ok], rp[ok]), "macro_f1": f1_score(y_te[ok], rp[ok], average="macro"),
                    "coverage": ok.mean()})
        # confronto alla pari: ogni metodo valutato sullo stesso sottoinsieme dove le regole si applicano
        for m, pr in list(preds.items()):
            res.append({"target": target, "scheme": scheme, "rep": rep, "method": f"{m} [rule subset]",
                        "accuracy": accuracy_score(y_te[ok], pr[ok]),
                        "macro_f1": f1_score(y_te[ok], pr[ok], average="macro"), "coverage": ok.mean()})
        fill = np.where(ok, rp.values, te.gang.map(gm).fillna(mode).values)
        add("Rules + GangMode", fill)
    return res


def main():
    full = prepare()
    out = []
    for target in os.environ.get("IMP_TARGETS", "country30,sector").split(","):
        for scheme in os.environ.get("IMP_SCHEMES", "MCAR,pattern").split(","):
            for rep in range(REPS):
                out += run_rep(full, target, scheme, rep)
                print(target, scheme, rep, flush=True)
    R = pd.DataFrame(out)
    R.to_csv(ROOT / "benchmark" / "results" / os.environ.get("IMPUTATION_OUT", "imputation.csv"), index=False)
    print(R.groupby(["target", "scheme", "method"])[["accuracy", "macro_f1", "coverage"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
