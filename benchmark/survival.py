#!/usr/bin/env python3
"""Benchmark 6 — sopravvivenza delle gang (quanto resta attivo un gruppo).

Durata di vita: dalla prima all'ultima rivendicazione. Una gang e' "cessata"
(evento) se non rivendica nulla negli ultimi INACTIVE giorni del dataset,
altrimenti e' censurata (ancora attiva al 30/09/2026).

Approccio a punto di riferimento (landmark) per evitare fughe di informazione:
si considerano le gang ancora attive LANDMARK giorni dopo la prima
rivendicazione; le feature usano solo quei primi LANDMARK giorni; si prevede
la durata RESIDUA.
Feature: rivendicazioni nei primi 60 giorni, intervallo mediano, quota di
vittime USA, numero di paesi e di settori, entropia dei settori, quota di
eventi visti da piu' fonti, anno di inizio.
Modelli: Kaplan-Meier per coorte di anno (descrittivo); Cox PH (lifelines,
penalizzato); Random Survival Forest e Gradient Boosting Survival
(scikit-survival); baseline = solo anno di inizio (Cox).
Metriche: indice di concordanza di Harrell e Brier integrato, con
validazione incrociata stratificata a 5 fold, 3 ripetizioni.
Uscita: benchmark/results/survival.csv, survival_km.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from sksurv.util import Surv
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
END, INACTIVE, LANDMARK = pd.Timestamp("2026-09-30"), 90, 60


def gang_table(version="enriched"):
    ev = pd.read_json(ROOT / "release" / f"events_{version}.json")
    ev["d"] = pd.to_datetime(ev.discovered)
    rows = []
    for g, x in ev.groupby("gang"):
        x = x.sort_values("d"); first, last = x.d.iloc[0], x.d.iloc[-1]
        life = (last - first).days
        if life < LANDMARK:
            continue
        w = x[x.d < first + pd.Timedelta(days=LANDMARK)]
        sec = w.sector.dropna().value_counts(normalize=True)
        rows.append({"gang": g, "start_year": first.year,
                     "duration": life - LANDMARK + 1, "event": int((END - last).days > INACTIVE),
                     "n_claims": len(w), "median_gap": w.d.diff().dt.days.median() if len(w) > 1 else LANDMARK,
                     "us_share": (w.country == "US").mean(), "n_countries": w.country.nunique(),
                     "n_sectors": w.sector.nunique(), "sector_entropy": float(-(sec * np.log(sec)).sum()) if len(sec) else 0.0,
                     "multi_source": (w.n_sources > 1).mean()})
    return pd.DataFrame(rows).fillna(0)


def main():
    out_dir = ROOT / "benchmark" / "results"
    G = gang_table()
    print("gang:", len(G), "cessate:", int(G.event.sum()), "censurate:", int((1 - G.event).sum()))
    km = []
    for coh, g in G.groupby(observed=True, by=pd.cut(G.start_year, [2017, 2021, 2022, 2023, 2024, 2026], labels=["<=2021", "2022", "2023", "2024", "2025-26"])):
        if len(g) < 5:
            continue
        k = KaplanMeierFitter().fit(g.duration, g.event)
        km.append({"cohort": str(coh), "n": len(g), "median_residual_days": k.median_survival_time_,
                   "surv_180d": float(k.predict(180)), "surv_365d": float(k.predict(365))})
    pd.DataFrame(km).to_csv(out_dir / "survival_km.csv", index=False)
    print(pd.DataFrame(km).round(3).to_string(index=False))

    feats = ["start_year", "n_claims", "median_gap", "us_share", "n_countries", "n_sectors", "sector_entropy", "multi_source"]
    res = []
    for rep in range(3):
        skf = StratifiedKFold(5, shuffle=True, random_state=rep)
        for fold, (tr, te) in enumerate(skf.split(G, G.event)):
            a, b = G.iloc[tr], G.iloc[te]
            ya, yb = Surv.from_arrays(a.event.astype(bool), a.duration), Surv.from_arrays(b.event.astype(bool), b.duration)
            t_grid = np.percentile(b.duration, [10, 30, 50, 70, 85])
            t_grid = t_grid[(t_grid > a.duration.min()) & (t_grid < min(a.duration.max(), b.duration.max()))]
            preds = {}
            for name, cols in (("Cox (start year only)", ["start_year"]), ("Cox PH", feats)):
                cph = CoxPHFitter(penalizer=0.1).fit(a[cols + ["duration", "event"]], "duration", "event")
                risk = cph.predict_partial_hazard(b[cols]).values
                sf = cph.predict_survival_function(b[cols], times=t_grid).T.values
                preds[name] = (risk, sf)
            for name, M in (("Random Survival Forest", RandomSurvivalForest(n_estimators=300, min_samples_leaf=5, random_state=rep, n_jobs=4)),
                            ("Gradient Boosting Survival", GradientBoostingSurvivalAnalysis(n_estimators=200, learning_rate=0.05, random_state=rep))):
                M.fit(a[feats], ya)
                sf = np.vstack([f(t_grid) for f in M.predict_survival_function(b[feats])])
                preds[name] = (M.predict(b[feats]), sf)
            for name, (risk, sf) in preds.items():
                c = concordance_index_censored(b.event.astype(bool), b.duration, risk)[0]
                # Brier integrato solo sui casi di test dentro l'intervallo osservato nel training
                ok = (b.duration < a.duration.max()).values
                ibs = (integrated_brier_score(ya, Surv.from_arrays(b.event[ok].astype(bool), b.duration[ok]), sf[ok], t_grid)
                       if len(t_grid) > 1 and ok.sum() > 5 else np.nan)
                res.append({"rep": rep, "fold": fold, "model": name, "c_index": c, "ibs": ibs})
    R = pd.DataFrame(res)
    R.to_csv(out_dir / "survival.csv", index=False)
    print(R.groupby("model")[["c_index", "ibs"]].agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
