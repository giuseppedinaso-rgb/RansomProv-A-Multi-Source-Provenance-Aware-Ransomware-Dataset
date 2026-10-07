#!/usr/bin/env python3
"""Sweep degli iperparametri per i modelli del compito 4 (prossimo evento).

Uso: python nextevent_sweep.py <n_trials> <modello> [<modello> ...]
  modelli: gru transformer tabformer flat mamba
Protocollo (come gli altri sweep):
  * ricerca bayesiana Optuna (TPE, seed 0) sulla versione enriched;
  * obiettivo = cross-entropy media di VALIDAZIONE (primo semestre 2025),
    la stessa usata per l'early stopping; il test non entra nella scelta;
  * spazio: learning rate, dimensione nascosta, strati, dropout, weight decay,
    lunghezza della finestra di contesto, variante della loss di addestramento
    (cross-entropy, label smoothing 0.1, pesata per classe con peso 1/sqrt(freq));
  * poi la configurazione migliore su 3 seed e su entrambe le versioni
    (risultati con suffisso "-tuned").
Studi: benchmark/results/nextevent_sweep/studies.db
"""
import sys, json
from pathlib import Path
import optuna
import nextevent_ptls as N

OUT = N.ROOT / "benchmark" / "results" / "nextevent_sweep"


def space(t):
    return {"lr": t.suggest_float("lr", 2e-4, 3e-3, log=True),
            "hidden": t.suggest_categorical("hidden", [64, 128, 256]),
            "layers": t.suggest_int("layers", 1, 3),
            "dropout": t.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3]),
            "weight_decay": t.suggest_categorical("weight_decay", [0.0, 0.01, 0.1]),
            "win": t.suggest_categorical("win", [32, 64, 128]),
            "loss": t.suggest_categorical("loss", ["ce", "smooth", "weighted"])}


def main():
    n, kinds = int(sys.argv[1]), sys.argv[2:]
    OUT.mkdir(parents=True, exist_ok=True)
    for kind in kinds:
        st = optuna.create_study(study_name=kind, direction="minimize", storage=f"sqlite:///{OUT / 'studies.db'}",
                                 load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=0))
        def obj(t):
            r = N.train_eval(kind, "enriched", 0, space(t), save=False)
            t.set_user_attr("val", r["val"]); return r["best_val_loss"]
        done = len([t for t in st.trials if t.state.is_finished()])
        if done < n:
            st.optimize(obj, n_trials=n - done, catch=(Exception,))
        best = st.best_trial.params
        json.dump({"model": kind, "best_val_loss": st.best_value, "params": best}, open(OUT / f"best_{kind}.json", "w"), indent=2)
        print(kind, "migliore", round(st.best_value, 4), best, flush=True)
        for seed in (0, 1, 2):
            for version in ("enriched", "observed"):
                r = N.train_eval(kind, version, seed, best, tag="-tuned")
                print(kind, version, seed, {f: round(r["test"][f]["weighted_f1"], 4) for f in N.FIELDS}, flush=True)


if __name__ == "__main__":
    main()
