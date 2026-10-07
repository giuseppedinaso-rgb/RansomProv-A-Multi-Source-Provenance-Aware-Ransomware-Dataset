#!/usr/bin/env python3
"""Test 1, parte C con GRU ottimizzato (configurazione migliore dello sweep del
compito 4): per ogni fonte, addestramento sulla vista a fonte singola e sul
dataset unito, valutati sugli STESSI eventi di test (quelli visti dalla fonte).
Versione observed (le viste a fonte singola non hanno imputazioni). 3 seed.
Uscita: benchmark/results/single_source_gru.json
"""
import json
import numpy as np
import nextevent_ptls as N
from nextevent_data import load, ROOT

best = json.load(open(ROOT / "benchmark" / "results" / "nextevent_sweep" / "best_gru.json"))["params"]
out = {}
for s in ("ransomlive", "ransomfeed", "ransomlook"):
    ev, _ = load("observed", s)
    ids = set(ev.event_id.values[ev.split.values == "test"])
    out[s] = {}
    for view, src in (("single_source", s), ("merged_history", None)):
        runs = [N.train_eval("gru", "observed", seed, best, save=False, source=src, eval_ids=ids)["test"] for seed in (0, 1, 2)]
        out[s][view] = {f: {m: float(np.mean([r[f][m] for r in runs])) for m in ("accuracy", "weighted_f1", "log_loss", "top3")}
                        for f in N.FIELDS}
        print(s, view, {f: round(out[s][view][f]["log_loss"], 3) for f in N.FIELDS}, flush=True)
json.dump(out, open(ROOT / "benchmark" / "results" / "single_source_gru.json", "w"), indent=2)
