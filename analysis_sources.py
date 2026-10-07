#!/usr/bin/env python3
"""Analisi delle fonti -> build/report_sources.json e figures/.

1. Sovrapposizione: eventi per combinazione di fonti, totale e per anno
   (figura stile UpSet).
2. Capture-recapture (stima dei sistemi multipli): modelli log-lineari di
   Poisson sulle 7 celle osservate della tabella 2x2x2 (fonti presenti /
   assenti), per anno, dal 2023 (prima ransomfeed copre poco: l'ipotesi di
   popolazione comune non regge). Modelli: indipendenza e tutte le
   combinazioni di interazioni a due vie (le fonti NON sono indipendenti:
   condividono origini e metodi di raccolta). Si riportano tutti i modelli con
   AIC e BIC e la stima della cella non osservata (eventi che nessuna delle tre
   fonti ha visto), con intervallo di confidenza profilo approssimato via
   bootstrap parametrico.
   Limiti: popolazione chiusa per anno; eterogeneita' di catturabilita' tra
   gang non modellata; un errore di abbinamento tra fonti (duplicato perso)
   gonfia la stima. Le stime sono un ordine di grandezza, non un conteggio.
3. Ritardo di pubblicazione: per gli eventi visti da piu' fonti, differenza in
   giorni tra la prima comparsa in ciascuna fonte; quale fonte arriva prima.
"""
import json, itertools
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BUILD, FIG = Path("build"), Path("figures")
SRC = ["ransomfeed", "ransomlive", "ransomlook"]
SHORT = {"ransomfeed": "RF", "ransomlive": "RL", "ransomlook": "LK"}


def overlap(ev):
    ev["combo"] = ev.sources.map(lambda s: "+".join(SHORT[x] for x in SRC if x in s))
    tot = ev.combo.value_counts().to_dict()
    by_year = pd.crosstab(ev.year, ev.combo)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    s = ev.combo.value_counts()
    ax.bar(s.index, s.values, color="#4C72B0")
    for i, v in enumerate(s.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("events"); ax.set_title("Events by combination of sources (RF=ransomfeed, RL=ransomware.live, LK=RansomLook)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "overlap.png", dpi=200); plt.close(fig)
    return {"total": tot, "by_year": {int(y): r[r > 0].to_dict() for y, r in by_year.iterrows()}}


def mse_year(sub):
    cells = list(itertools.product([0, 1], repeat=3))[1:]          # 7 celle osservate
    X = pd.DataFrame(cells, columns=["A", "B", "C"])
    cnt = {c: 0 for c in cells}
    for s in sub.sources:
        cnt[tuple(int(x in s) for x in SRC)] += 1
    y = np.array([cnt[c] for c in cells])
    inter = {"AB": X.A * X.B, "AC": X.A * X.C, "BC": X.B * X.C}
    out = []
    for k in range(0, 3):
        for terms in itertools.combinations(inter, k):
            Z = sm.add_constant(pd.concat([X] + [inter[t].rename(t) for t in terms], axis=1).astype(float))
            m = sm.GLM(y, Z, family=sm.families.Poisson()).fit()
            n0 = float(np.exp(m.params["const"]))
            # bootstrap parametrico per l'intervallo
            rng = np.random.default_rng(0); bs = []
            for _ in range(300):
                yb = rng.poisson(m.fittedvalues)
                try:
                    bs.append(float(np.exp(sm.GLM(yb, Z, family=sm.families.Poisson()).fit().params["const"])))
                except Exception:
                    pass
            out.append({"model": "indep" if not terms else "+".join(terms), "aic": round(m.aic, 1),
                        "bic": round(m.bic_llf, 1) if hasattr(m, "bic_llf") else None, "df_resid": int(m.df_resid),
                        "unseen": round(n0), "unseen_ci95": [round(np.percentile(bs, 2.5)), round(np.percentile(bs, 97.5))],
                        "observed": int(y.sum()), "total_est": round(y.sum() + n0)})
    return sorted(out, key=lambda r: r["aic"])


def delays(u, r):
    d = u.merge(r, on=["source", "source_id"])
    first = d.groupby(["event_id", "source"]).discovered.min().unstack()
    res = {}
    for a, b in itertools.combinations(SRC, 2):
        x = (first[b] - first[a]).dt.days.dropna()
        res[f"{SHORT[b]}-{SHORT[a]}"] = {"n": int(len(x)), "median": float(x.median()),
                                          "iqr": [float(x.quantile(.25)), float(x.quantile(.75))],
                                          "same_day_pct": round((x == 0).mean() * 100, 1),
                                          "within_1d_pct": round((x.abs() <= 1).mean() * 100, 1)}
    multi = first.dropna(thresh=2)
    mn = multi.min(axis=1)
    res["first_source_pct"] = {SHORT[s]: round(((multi[s] == mn)).mean() * 100, 1) for s in SRC}
    return res


def main():
    FIG.mkdir(exist_ok=True)
    ev = pd.read_json("release/events_enriched.json")
    ev["year"] = pd.to_datetime(ev.discovered).dt.year
    rep = {"overlap": overlap(ev)}
    rep["capture_recapture"] = {int(y): mse_year(g) for y, g in ev[ev.year >= 2023].groupby("year")}
    r = pd.read_json(BUILD / "record_to_event.json", dtype={"source_id": str})
    r = r[r.source_id.notna() & (r.source_id != "None")].drop_duplicates(["source", "source_id"])
    u = pd.read_json(BUILD / "unified.json", dtype={"source_id": str}, convert_dates=["discovered"])
    rep["delays_days"] = delays(u, r)
    json.dump(rep, open(BUILD / "report_sources.json", "w"), indent=2)
    print(json.dumps(rep["overlap"]["total"], indent=1))
    for y, ms in rep["capture_recapture"].items():
        print(y, [(m["model"], m["aic"], m["unseen"], m["unseen_ci95"]) for m in ms[:4]])
    print(json.dumps(rep["delays_days"], indent=1))


if __name__ == "__main__":
    main()
