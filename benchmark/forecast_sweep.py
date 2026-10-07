#!/usr/bin/env python3
"""Benchmark 1b — sweep dei modelli neurali di previsione (neuralforecast Auto*).

Per ogni modello (NHITS, NBEATS, TFT, PatchTST, TiDE, LSTM) e versione:
  1. ricerca degli iperparametri con Optuna (spazio di default di
     neuralforecast, NUM_SAMPLES prove) addestrando sui dati fino al
     30/06/2025 e validando sulle ultime 26 settimane di quel periodo
     (1/1-30/6/2025): il test non entra nella scelta;
  2. con la configurazione migliore, la stessa valutazione a origine mobile di
     forecast.py (16 finestre da 4 settimane, luglio 2025 - settembre 2026).
Modelli di base: Chronos-Bolt tiny/mini/small/base zero-shot (nessun tuning).
Uscita: benchmark/results/forecast_sweep_summary.csv e best_configs.json
"""
import os, sys, json, warnings, logging
from pathlib import Path
import pandas as pd
import optuna
from forecast import build_series, windows, evaluate, H, CUTOFF

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark" / "results"
NUM_SAMPLES = 20


def tune_and_eval(df, origins, name):
    from neuralforecast import NeuralForecast
    from neuralforecast import auto, models
    Auto, Base = getattr(auto, f"Auto{name}"), getattr(models, name)
    tr = df[df.ds <= CUTOFF][["unique_id", "ds", "y"]]
    base_cfg = Auto.get_default_config(h=H, backend="optuna")
    # start_padding_enabled: le serie per gang iniziano tardi e alcune sono piu' corte della finestra d'ingresso
    def cfg_fn(trial):
        c = {**base_cfg(trial), "start_padding_enabled": True}
        if name == "LSTM":
            # lo spazio di default di AutoLSTM arriva a lr 1e-1: sulla versione
            # observed l'addestramento divergeva (loss NaN). Spazio ristretto e
            # gradient clipping, applicati a entrambe le versioni.
            c["learning_rate"] = trial.suggest_float("lstm_lr", 1e-4, 1e-2, log=True)
            c["gradient_clip_val"] = 1.0
        return c
    am = Auto(h=H, num_samples=NUM_SAMPLES, backend="optuna", config=cfg_fn,
              search_alg=optuna.samplers.TPESampler(seed=0))
    nf0 = NeuralForecast(models=[am], freq="W-MON")
    nf0.fit(tr, val_size=26)
    best = nf0.models[0].results.best_trial.user_attrs["ALL_PARAMS"]    # i risultati stanno nella copia interna
    cfg = {k: v for k, v in best.items() if k not in ("h", "loss", "valid_loss")}
    cfg.update(enable_progress_bar=False, logger=False, enable_checkpointing=False)
    res = []
    for o in origins:
        t = df[df.ds < o][["unique_id", "ds", "y"]]
        nf = NeuralForecast(models=[Base(h=H, **cfg)], freq="W-MON")
        nf.fit(t)
        p = nf.predict().rename(columns={name: f"{name} (tuned)"})
        res.append(p.reset_index() if "unique_id" not in p.columns else p)
    clean = {k: (v if isinstance(v, (int, float, str, bool, list, type(None))) else str(v)) for k, v in best.items()}
    return pd.concat(res), clean


def run_chronos(df, origins, size):
    import torch
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(f"amazon/chronos-bolt-{size}", device_map="cuda", torch_dtype=torch.float32)
    res = []
    for o in origins:
        tr = df[df.ds < o]; ids = sorted(tr.unique_id.unique())
        ctx = [torch.tensor(tr[tr.unique_id == i].sort_values("ds").y.values, dtype=torch.float32) for i in ids]
        _, mean = pipe.predict_quantiles(ctx, prediction_length=H, quantile_levels=[0.5])
        fut = pd.date_range(o, periods=H, freq="W-MON")
        for i, uid in enumerate(ids):
            res.append(pd.DataFrame({"unique_id": uid, "ds": fut, f"Chronos-Bolt-{size}": mean[i].numpy()}))
    return pd.concat(res)


def main():
    names = sys.argv[1:] or ["NHITS", "NBEATS", "TFT", "PatchTST", "TiDE", "LSTM"]
    best_all, sums = {}, []
    for version in ("enriched", "observed"):
        df = build_series(version); origins = windows(df)
        fcs = []
        if os.environ.get("SKIP_CHRONOS") is None:
            for size in ("tiny", "mini", "small", "base"):
                fcs.append(run_chronos(df, origins, size).set_index(["unique_id", "ds"]))
                print(version, "chronos", size, "ok", flush=True)
        for n in names:
            try:
                f, best = tune_and_eval(df, origins, n)
                fcs.append(f.set_index(["unique_id", "ds"])); best_all[f"{n}_{version}"] = best
                print(version, n, "ok", flush=True)
            except Exception as e:
                print(version, n, "ERRORE", type(e).__name__, str(e)[:300], flush=True)
        fc = pd.concat(fcs, axis=1).reset_index()
        per, summ = evaluate(df, fc, version)
        per.to_csv(OUT / f"forecast_sweep_per_series_{version}.csv", index=False); sums.append(summ)
        json.dump(best_all, open(OUT / "forecast_sweep_best_configs.json", "w"), indent=2, default=str)
    S = pd.concat(sums)
    prev = OUT / "forecast_sweep_summary.csv"
    if prev.exists():                                   # tiene i risultati Chronos gia' calcolati
        P = pd.read_csv(prev); S = pd.concat([P[~P.model.isin(S.model.unique())], S])
    S.to_csv(prev, index=False)
    print(S.pivot_table(index=["group", "model"], columns="version", values="MASE").round(3).to_string())


if __name__ == "__main__":
    main()
