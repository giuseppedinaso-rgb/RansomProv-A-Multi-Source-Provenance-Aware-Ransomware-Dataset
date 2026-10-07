#!/usr/bin/env python3
"""Benchmark 1c — baseline per serie intermittenti e riconciliazione gerarchica.

1. Serie intermittenti (molte settimane a zero, tipiche delle gang): Croston
   classico, Croston-SBA (Syntetos & Boylan), TSB (Teunter-Syntetos-Babai),
   ADIDA, IMAPA (statsforecast).
2. Riconciliazione gerarchica: per ciascuna dimensione (paese, settore, gang)
   la gerarchia e' totale -> categorie (le piu' frequenti, 'other' e
   'unknown' per gli eventi senza valore), cosi' che le foglie sommino al
   totale. Previsioni di base AutoETS e AutoTheta su tutti i nodi, riconciliate
   con BottomUp, TopDown (proporzioni previste), MinTrace OLS e MinTrace
   mint_shrink (Wickramasuriya et al. 2019) con hierarchicalforecast.
Stessa valutazione di forecast.py: origine mobile, 16 finestre da 4
settimane (luglio 2025 - settembre 2026), MASE per serie.
Uscita: benchmark/results/forecast_hier_summary.csv
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, AutoTheta, CrostonClassic, CrostonSBA, TSB, ADIDA, IMAPA, Naive
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp, TopDown, MinTrace
from hierarchicalforecast.utils import aggregate
from forecast import START, END, CUTOFF, H, windows

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark" / "results"
TOPK = {"country": 10, "sector": 13, "gang": 20}


def bottom_df(ev, dim):
    ev = ev[(ev.d >= START) & (ev.d <= END)].copy()
    train = ev[ev.d <= CUTOFF]
    if dim == "gang":
        top = train[train.d >= "2024-01-01"].gang.value_counts().head(TOPK[dim]).index
    else:
        top = train[dim].value_counts().head(TOPK[dim]).index
    v = ev[dim].where(ev[dim].isin(top), "other").where(ev[dim].notna(), "unknown")
    ev["cat"] = f"{dim}:" + v.astype(str)
    ev["ds"] = ev.d.dt.to_period("W-SUN").dt.start_time
    g = ev.groupby(["cat", "ds"]).size().rename("y").reset_index()
    full = pd.MultiIndex.from_product([g.cat.unique(), pd.date_range(START, END, freq="W-MON")], names=["cat", "ds"])
    g = g.set_index(["cat", "ds"]).reindex(full, fill_value=0).reset_index()
    g["total"] = "total"; g["y"] = g.y.astype(float)
    return g


def mase(df_true, fc, models, scale):
    m = fc.merge(df_true, on=["unique_id", "ds"])
    rows = []
    for uid, g in m.groupby("unique_id"):
        for mod in models:
            mae = (g[mod].clip(lower=0) - g.y).abs().mean()
            rows.append({"unique_id": uid, "model": mod, "MASE": mae / scale.get(uid, np.nan)})
    return pd.DataFrame(rows)


def run(version, dim):
    ev = pd.read_json(ROOT / "release" / f"events_{version}.json"); ev["d"] = pd.to_datetime(ev.discovered)
    B = bottom_df(ev, dim)
    Y, S, tags = aggregate(B, [["total"], ["total", "cat"]])
    Y = Y.reset_index() if "unique_id" not in Y.columns else Y
    scale = {u: g.sort_values("ds").y.diff().abs().mean() for u, g in Y[Y.ds <= CUTOFF].groupby("unique_id")}
    origins = windows(Y.assign(group=dim))
    base_models = [AutoETS(season_length=1), AutoTheta(season_length=1)]
    inter_models = [CrostonClassic(), CrostonSBA(), TSB(alpha_d=0.2, alpha_p=0.2), ADIDA(), IMAPA(), Naive()]
    allfc = []
    for o in origins:
        tr = Y[Y.ds < o]
        sf = StatsForecast(models=base_models, freq="W-MON", n_jobs=4)
        fc = sf.forecast(df=tr, h=H, fitted=True)
        fitted = sf.forecast_fitted_values()
        rec = HierarchicalReconciliation([BottomUp(), TopDown(method="forecast_proportions"),
                                          MinTrace(method="ols"), MinTrace(method="mint_shrink")])
        r = rec.reconcile(Y_hat_df=fc, Y_df=fitted, S_df=S, tags=tags)
        r = r.reset_index() if "unique_id" not in r.columns else r
        f2 = StatsForecast(models=inter_models, freq="W-MON", n_jobs=4).forecast(df=tr, h=H)
        f2 = f2.reset_index() if "unique_id" not in f2.columns else f2
        allfc.append(r.merge(f2, on=["unique_id", "ds"]))
    fc = pd.concat(allfc)
    models = [c for c in fc.columns if c not in ("unique_id", "ds")]
    per = mase(Y, fc, models, scale)
    per["level"] = np.where(per.unique_id == "total", "total", "bottom")
    per["version"], per["dimension"] = version, dim
    return per


def main():
    per = pd.concat([run(v, d) for v in ("enriched", "observed") for d in ("country", "sector", "gang")])
    per.to_csv(OUT / "forecast_hier_per_series.csv", index=False)
    S = per.groupby(["dimension", "level", "model", "version"]).MASE.mean().round(3).unstack("version").reset_index()
    S.to_csv(OUT / "forecast_hier_summary.csv", index=False)
    print(S.to_string(index=False))


if __name__ == "__main__":
    main()
