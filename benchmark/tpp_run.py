#!/usr/bin/env python3
"""Benchmark 3 — addestra i modelli EasyTPP e raccoglie i risultati.

Uso: python tpp_run.py <gpu> <modello> [<modello> ...]
  modelli: RMTPP NHP THP SAHP IntensityFree AttNHP S2P2
Per ogni (versione, mark) genera la configurazione, addestra con early stopping
implicito (EasyTPP salva il modello con la miglior log-likelihood di
validazione) e registra le metriche di TEST all'epoca migliore di validazione:
log-likelihood per evento (EasyTPP la riporta gia' mediata sugli eventi),
accuratezza sul tipo, RMSE sul tempo (giorni).

Nota sul confronto observed/enriched: la log-likelihood e l'accuratezza
dipendono dalla quota di eventi 'unknown' (diversa tra le due versioni); la
RMSE sul tempo dipende solo dai tempi ed e' direttamente confrontabile.

Ambiente: Python 3.10 con easy_tpp 0.3.0. Se le librerie cuDNN di sistema sono in
conflitto con quelle di PyTorch, lanciare con LD_LIBRARY_PATH="".
"""
import sys, re, json, subprocess, os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA, RES = ROOT / "benchmark" / "tpp_data", ROOT / "benchmark" / "results" / "tpp"
EPOCHS = {"RMTPP": 200, "NHP": 100, "THP": 100, "SAHP": 100, "IntensityFree": 200, "AttNHP": 40, "S2P2": 100}
MODEL_CFG = {
    "RMTPP": dict(hidden_size=32, time_emb_size=16, num_layers=2, num_heads=2),
    "NHP": dict(hidden_size=64),
    "THP": dict(hidden_size=64, time_emb_size=16, num_layers=2, num_heads=2),
    "SAHP": dict(hidden_size=64, time_emb_size=16, num_layers=2, num_heads=2),
    "IntensityFree": dict(hidden_size=64, num_mix_components=16),
    "AttNHP": dict(hidden_size=32, time_emb_size=16, num_layers=1, num_heads=2),
    "S2P2": dict(hidden_size=64, num_layers=2),
}
# parametri che EasyTPP legge da model_config.model_specs
MODEL_SPECS = {"IntensityFree": dict(num_mix_components=16), "S2P2": dict(P=32)}
THINNING = dict(num_seq=2, num_sample=1, num_exp=50, look_ahead_time=10, patience_counter=5,
                over_sample_rate=5, num_samples_boundary=5, dtime_max=30)


def config(version, mark, model, gpu, params=None, tag=""):
    """params: iperparametri dello sweep (sovrascrivono i default). Chiavi:
    lr, batch_size, max_epoch, seed, dropout, e qualunque chiave di model_config;
    'spec_<nome>' va in model_specs."""
    params = dict(params or {})
    d = DATA / f"{version}_{mark}"
    k = len((d / "vocab.txt").read_text().splitlines())
    ds, ex = f"{version}_{mark}", f"{model}_{version}_{mark}{tag}"
    mc = {**MODEL_CFG[model], "dropout": 0.1, "use_ln": False, "loss_integral_num_sample_per_step": 20,
          "mc_num_sample_per_step": 20, "sharing_param_layer": False, "thinning": THINNING}
    specs = dict(MODEL_SPECS.get(model, {}))
    tc = {"batch_size": 16, "max_epoch": EPOCHS[model], "learning_rate": 1e-3, "seed": 2019}
    for key, v in params.items():
        if key.startswith("spec_"):
            specs[key[5:]] = v
        elif key == "lr":
            tc["learning_rate"] = v
        elif key in ("batch_size", "max_epoch", "seed"):
            tc[key] = v
        else:
            mc[key] = v
    if specs:
        mc["model_specs"] = specs
    return {
        "pipeline_config_id": "runner_config",
        "data": {ds: {"data_format": "pkl", "train_dir": str(d / "train.pkl"), "valid_dir": str(d / "dev.pkl"),
                      "test_dir": str(d / "test.pkl"),
                      "data_specs": {"num_event_types": k, "pad_token_id": k, "padding_side": "right",
                                     "truncation_side": "right"}}},
        ex: {"base_config": {"stage": "train", "backend": "torch", "dataset_id": ds, "runner_id": "std_tpp",
                             "model_id": model, "base_dir": str(RES / "checkpoints") + "/"},
             "trainer_config": {**tc, "shuffle": True, "optimizer": "adam", "valid_freq": 1, "use_tfb": False,
                                "metrics": ["acc", "rmse"], "gpu": gpu},
             "model_config": mc},
    }, ex


def parse(log):
    valid, test = {}, {}
    for line in log.splitlines():
        m = re.search(r"\[ Epoch (\d+) \((valid|test)\) \]:\s+\w+ (.*)", line)
        if m:
            ep, ph, rest = int(m.group(1)), m.group(2), m.group(3)
            vals = dict((a.strip(), float(b)) for a, b in re.findall(r"(\w+) is ([-\d.eE+]+)", rest))
            (valid if ph == "valid" else test)[ep] = vals
    if not valid:
        return None
    best = max(valid, key=lambda e: valid[e].get("loglike", float("-inf")))
    return {"best_epoch": best, "valid": valid[best], "test": test.get(best, {})}


def train_one(cpath, ex):
    from easy_tpp.config_factory import Config
    from easy_tpp.runner import Runner
    Runner.build_from_config(Config.build_from_yaml_file(cpath, experiment_id=ex)).run()


def run_config(cfg, ex, cpath):
    """Esegue una corsa in un sottoprocesso; restituisce (metriche, log)."""
    yaml.safe_dump(cfg, open(cpath, "w"))
    p = subprocess.run([sys.executable, __file__, "--one", str(cpath), ex], capture_output=True,
                       text=True, env={**os.environ, "LD_LIBRARY_PATH": ""})
    log = p.stdout + p.stderr
    return parse(log), log


def main():
    if sys.argv[1] == "--one":                       # sottoprocesso: una sola corsa
        return train_one(sys.argv[2], sys.argv[3])
    gpu, models = int(sys.argv[1]), sys.argv[2:]
    RES.mkdir(parents=True, exist_ok=True)
    for model in models:
        for version in ("observed", "enriched"):
            for mark in ("sector", "country"):
                out = RES / f"{model}_{version}_{mark}.json"
                if out.exists():
                    continue
                cfg, ex = config(version, mark, model, gpu)
                cpath = RES / f"{ex}.yaml"
                yaml.safe_dump(cfg, open(cpath, "w"))
                p = subprocess.run([sys.executable, __file__, "--one", str(cpath), ex], capture_output=True,
                                   text=True, env={**os.environ, "LD_LIBRARY_PATH": ""})
                log = p.stdout + p.stderr
                (RES / f"{ex}.log").write_text(log)
                res = parse(log) or {"error": log[-500:]}
                json.dump({"model": model, "version": version, "mark": mark, **res}, open(out, "w"), indent=2)
                print(ex, json.dumps(res)[:300], flush=True)


if __name__ == "__main__":
    main()
