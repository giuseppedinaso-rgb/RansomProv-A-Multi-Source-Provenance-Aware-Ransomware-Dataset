#!/usr/bin/env python3
"""Benchmark 1 — previsione dei conteggi settimanali di rivendicazioni.

Serie: totale; per settore (13); per paese (10 piu' frequenti + altri); per gang
(20 piu' attive nel periodo di training). Ogni serie e' costruita sia da
events_observed sia da events_enriched: per paese e settore le due versioni
differiscono (nella prima gli eventi senza valore sono esclusi).

Valutazione: origine mobile (rolling origin). Training fino al 30/06/2025,
poi finestre successive di h=4 settimane fino a fine settembre 2026; ogni
finestra usa tutta la storia fino alla sua origine. Dal 2021 (prima ci sono
troppi pochi eventi per serie settimanali).

Modelli:
  statistici (statsforecast)  Naive, SeasonalNaive(4), HistoricAverage, AutoETS, AutoARIMA, AutoTheta
  ML globale (mlforecast)     LightGBM con ritardi 1-4, 8, 12, 26, 52 e medie mobili
  rete neurale (neuralforecast) NHITS, globale su tutte le serie
  modello di base zero-shot   Chronos-Bolt (amazon/chronos-bolt-base), nessun addestramento

Metriche: MAE, e MASE (MAE scalato sull'errore in-sample del naive a 1 passo, per
serie), aggregati per gruppo di serie con media. Risultati in
benchmark/results/forecast_*.csv.
"""
import os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark" / "results"
START, CUTOFF, END = "2021-01-04", "2025-06-30", "2026-09-27"
H = 4


def weekly(ev, col=None, keep=None):
    ev = ev[(ev.d >= START) & (ev.d <= END)].copy()
    ev["ds"] = ev.d.dt.to_period("W-SUN").dt.start_time
    if col is None:
        ev["unique_id"] = "total"
    else:
        ev = ev[ev[col].notna()]
        ev["unique_id"] = ev[col].where(ev[col].isin(keep), "other") if keep is not None else ev[col]
        ev["unique_id"] = col + ":" + ev.unique_id.astype(str)
    df = ev.groupby(["unique_id", "ds"]).size().rename("y").reset_index()
    full = pd.MultiIndex.from_product([df.unique_id.unique(),
                                       pd.date_range(START, END, freq="W-MON")], names=["unique_id", "ds"])
    return df.set_index(["unique_id", "ds"]).reindex(full, fill_value=0).reset_index().astype({"y": float})


def build_series(version):
    ev = pd.read_json(ROOT / "release" / f"events_{version}.json")
    ev["d"] = pd.to_datetime(ev.discovered)
    train = ev[ev.d <= CUTOFF]
    top_c = train.country.value_counts().head(10).index
    top_g = train[train.d >= "2024-01-01"].gang.value_counts().head(20).index
    parts = {"total": weekly(ev), "sector": weekly(ev, "sector"), "country": weekly(ev, "country", top_c),
             "gang": weekly(ev, "gang", top_g)}
    return pd.concat([p.assign(group=g) for g, p in parts.items()])


def windows(df):
    cut = pd.Timestamp(CUTOFF)
    ds = sorted(df.ds.unique())
    origins = [d for d in ds if d > cut][::H]
    return [o for o in origins if o + pd.Timedelta(weeks=H - 1) <= pd.Timestamp(END)]


def run_stats(df, origins):
    from statsforecast import StatsForecast
    from statsforecast.models import Naive, SeasonalNaive, HistoricAverage, AutoETS, AutoARIMA, AutoTheta
    res = []
    for o in origins:
        tr = df[df.ds < o][["unique_id", "ds", "y"]]
        sf = StatsForecast(models=[Naive(), SeasonalNaive(season_length=4), HistoricAverage(),
                                   AutoETS(season_length=1), AutoARIMA(season_length=1), AutoTheta(season_length=1)],
                           freq="W-MON", n_jobs=8)
        res.append(sf.forecast(df=tr, h=H).reset_index(drop=False) if "unique_id" not in sf.forecast(df=tr, h=H).columns
                   else sf.forecast(df=tr, h=H))
    return pd.concat(res)


def run_lgbm(df, origins):
    import lightgbm as lgb
    from mlforecast import MLForecast
    from mlforecast.lag_transforms import RollingMean
    res = []
    for o in origins:
        tr = df[df.ds < o][["unique_id", "ds", "y"]]
        mlf = MLForecast(models={"LightGBM": lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, verbose=-1)},
                         freq="W-MON", lags=[1, 2, 3, 4, 8, 12, 26, 52],
                         lag_transforms={1: [RollingMean(4), RollingMean(12)]}, date_features=["month"])
        mlf.fit(tr)
        res.append(mlf.predict(H))
    return pd.concat(res)


def run_nhits(df, origins):
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS
    res = []
    for o in origins:
        tr = df[df.ds < o][["unique_id", "ds", "y"]]
        nf = NeuralForecast(models=[NHITS(h=H, input_size=26, max_steps=300, scaler_type="robust",
                                          random_seed=0, enable_progress_bar=False, logger=False)], freq="W-MON")
        nf.fit(tr)
        res.append(nf.predict())
    return pd.concat(res)


def run_chronos(df, origins):
    import torch
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-base", device_map="cuda", torch_dtype=torch.float32)
    res = []
    for o in origins:
        tr = df[df.ds < o]
        ids = sorted(tr.unique_id.unique())
        ctx = [torch.tensor(tr[tr.unique_id == i].sort_values("ds").y.values, dtype=torch.float32) for i in ids]
        _, mean = pipe.predict_quantiles(ctx, prediction_length=H, quantile_levels=[0.5])
        fut = pd.date_range(o, periods=H, freq="W-MON")
        for i, uid in enumerate(ids):
            res.append(pd.DataFrame({"unique_id": uid, "ds": fut, "Chronos-Bolt": mean[i].numpy()}))
    return pd.concat(res)


def evaluate(df, fc, version):
    models = [c for c in fc.columns if c not in ("unique_id", "ds")]
    m = fc.merge(df[["unique_id", "ds", "y", "group"]], on=["unique_id", "ds"])
    scale = {}
    for uid, g in df[df.ds <= CUTOFF].groupby("unique_id"):
        s = g.sort_values("ds").y.diff().abs().mean()
        scale[uid] = s if s and s > 0 else np.nan
    rows = []
    for (grp, uid), g in m.groupby(["group", "unique_id"]):
        for mod in models:
            mae = (g[mod].clip(lower=0) - g.y).abs().mean()
            rows.append({"version": version, "group": grp, "unique_id": uid, "model": mod,
                         "MAE": mae, "MASE": mae / scale.get(uid, np.nan)})
    per = pd.DataFrame(rows)
    summ = per.groupby(["version", "group", "model"])[["MAE", "MASE"]].mean().round(3).reset_index()
    return per, summ


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["stats", "lgbm", "nhits", "chronos"]
    allsum = []
    for version in ("observed", "enriched"):
        df = build_series(version)
        origins = windows(df)
        print(f"[{version}] serie {df.unique_id.nunique()} | finestre {len(origins)} x {H} settimane", flush=True)
        fcs = []
        for name, fn in (("stats", run_stats), ("lgbm", run_lgbm), ("nhits", run_nhits), ("chronos", run_chronos)):
            if name in which:
                f = fn(df, origins)
                fcs.append(f.set_index(["unique_id", "ds"]))
                print(f"  {name} ok", flush=True)
        fc = pd.concat(fcs, axis=1).reset_index()
        per, summ = evaluate(df, fc, version)
        per.to_csv(OUT / f"forecast_per_series_{version}.csv", index=False)
        allsum.append(summ)
    s = pd.concat(allsum)
    s.to_csv(OUT / "forecast_summary.csv", index=False)
    print(s.pivot_table(index=["group", "model"], columns="version", values="MASE").round(3).to_string())


if __name__ == "__main__":
    main()
