#!/usr/bin/env python3
"""Sweep degli iperparametri per i processi puntuali (EasyTPP) con Optuna.

Uso: python tpp_sweep.py <gpu> <mark> <n_trials> <modello> [<modello> ...]
     python tpp_sweep.py final <gpu> <mark> <modello> [...]   # seed finali sulla config migliore

Protocollo:
  * selezione SOLO sulla validazione (primo semestre 2025) della versione
    enriched: obiettivo = log-likelihood per evento all'epoca migliore;
    il test non entra mai nella scelta;
  * ricerca bayesiana (TPE, seed fisso), n_trials prove per modello e mark,
    con epoche ridotte (TRIAL_EPOCHS); una prova che diverge (NaN, nessuna
    metrica) vale -inf;
  * fase finale: la configurazione migliore, con le epoche piene, su 3 seed e
    su ENTRAMBE le versioni (observed ed enriched): lo stesso tuning per le due,
    cosi' il confronto non dipende dalla selezione.
Gli studi sono salvati in benchmark/results/tpp_sweep/studies.db (si riprendono).
"""
import sys, json, math
from pathlib import Path
import optuna
from tpp_run import config, run_config, EPOCHS, RES as _RES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark" / "results" / "tpp_sweep"
TRIAL_EPOCHS = {"RMTPP": 100, "IntensityFree": 100, "NHP": 60, "THP": 60, "SAHP": 60, "AttNHP": 25, "S2P2": 60}
SEEDS = [2019, 1, 2]


def space(trial, model):
    p = {"lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
         "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
         "dropout": trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3])}
    if model == "RMTPP":
        p.update(hidden_size=trial.suggest_categorical("hidden_size", [16, 32, 64, 128]),
                 time_emb_size=trial.suggest_categorical("time_emb_size", [8, 16, 32]))
    elif model == "NHP":
        p.update(hidden_size=trial.suggest_categorical("hidden_size", [32, 64, 128]))
    elif model in ("THP", "SAHP", "AttNHP"):
        p.update(hidden_size=trial.suggest_categorical("hidden_size", [32, 64, 128]),
                 num_layers=trial.suggest_int("num_layers", 1, 3 if model != "AttNHP" else 2),
                 num_heads=trial.suggest_categorical("num_heads", [1, 2, 4]),
                 time_emb_size=trial.suggest_categorical("time_emb_size", [8, 16, 32]),
                 use_ln=trial.suggest_categorical("use_ln", [False, True]))
    elif model == "IntensityFree":
        p.update(hidden_size=trial.suggest_categorical("hidden_size", [32, 64, 128]),
                 spec_num_mix_components=trial.suggest_categorical("num_mix_components", [4, 8, 16, 32]))
    elif model == "S2P2":
        p.update(hidden_size=trial.suggest_categorical("hidden_size", [32, 64, 128]),
                 num_layers=trial.suggest_int("num_layers", 1, 4),
                 spec_P=trial.suggest_categorical("P", [16, 32, 64]),
                 spec_dropout_rate=trial.suggest_categorical("ssm_dropout", [0.0, 0.1, 0.2]))
    return p


def sweep(gpu, mark, n_trials, model):
    OUT.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(study_name=f"{model}_{mark}", direction="maximize",
                                storage=f"sqlite:///{OUT / 'studies.db'}", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=0))

    def objective(trial):
        p = space(trial, model)
        p["max_epoch"] = TRIAL_EPOCHS[model]
        cfg, ex = config("enriched", mark, model, gpu, p, tag=f"_t{trial.number}")
        res, log = run_config(cfg, ex, OUT / f"{ex}.yaml")
        (OUT / f"{ex}.log").write_text(log[-20000:])
        ll = (res or {}).get("valid", {}).get("loglike")
        if ll is None or not math.isfinite(ll):
            return float("-inf")
        trial.set_user_attr("valid", res["valid"]); trial.set_user_attr("best_epoch", res["best_epoch"])
        return ll

    done = len([t for t in study.trials if t.state.is_finished()])
    if done < n_trials:
        study.optimize(objective, n_trials=n_trials - done)
    b = study.best_trial
    json.dump({"model": model, "mark": mark, "best_valid_loglike": b.value, "params": b.params,
               "n_trials": len(study.trials)}, open(OUT / f"best_{model}_{mark}.json", "w"), indent=2)
    print(model, mark, "migliore", round(b.value, 4), b.params, flush=True)


def final(gpu, mark, model):
    best = json.load(open(OUT / f"best_{model}_{mark}.json"))
    pr = dict(best["params"])
    p = {k: v for k, v in pr.items() if k not in ("num_mix_components", "P", "ssm_dropout")}
    if "num_mix_components" in pr: p["spec_num_mix_components"] = pr["num_mix_components"]
    if "P" in pr: p["spec_P"] = pr["P"]
    if "ssm_dropout" in pr: p["spec_dropout_rate"] = pr["ssm_dropout"]
    p["max_epoch"] = EPOCHS[model]
    for version in ("observed", "enriched"):
        for seed in SEEDS:
            out = OUT / f"final_{model}_{version}_{mark}_s{seed}.json"
            if out.exists():
                continue
            cfg, ex = config(version, mark, model, gpu, {**p, "seed": seed}, tag=f"_final_s{seed}")
            res, log = run_config(cfg, ex, OUT / f"{ex}.yaml")
            (OUT / f"{ex}.log").write_text(log[-20000:])
            json.dump({"model": model, "version": version, "mark": mark, "seed": seed, "params": p,
                       **(res or {"error": log[-500:]})}, open(out, "w"), indent=2)
            print(ex, json.dumps((res or {}).get("test", {}))[:200], flush=True)


def main():
    if sys.argv[1] == "final":
        gpu, mark = int(sys.argv[2]), sys.argv[3]
        for m in sys.argv[4:]:
            final(gpu, mark, m)
        return
    gpu, mark, n = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    for m in sys.argv[4:]:
        sweep(gpu, mark, n, m)
        final(gpu, mark, m)


if __name__ == "__main__":
    main()
