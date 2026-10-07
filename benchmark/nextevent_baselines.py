#!/usr/bin/env python3
"""Baseline statistiche per il protocollo nextevent_data (come MostPopular /
Last-K del benchmark HoTPP).
  MostPopular   classe piu' frequente nel training
  LastValue     valore dell'evento precedente della stessa gang
  GangMode      valore piu' frequente nella storia della gang (ultimi 50 eventi)
Le storie usano le etichette della versione (observed/enriched), mai il test.
Uso: python nextevent_baselines.py
"""
import json
from collections import Counter, deque
import numpy as np
from nextevent_data import load, targets, metrics, FIELDS, IGNORE, ROOT

OUT = ROOT / "benchmark" / "results" / "nextevent"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for version in ("observed", "enriched"):
        ev, vocab = load(version); tgt = targets(ev, vocab)
        feat = {"delta": ev.delta.values, "country": ev.country_i.values, "sector": ev.sector_i.values}
        unk = {"delta": 0, "country": vocab["country"].index("unknown"), "sector": vocab["sector"].index("unknown")}
        tr = ev.split.values == "train"
        res = {m: {"val": {}, "test": {}} for m in ("MostPopular", "LastValue", "GangMode")}
        res["GangMode"]["alpha"] = {}
        for f in FIELDS:
            y = tgt[f]; K = len(vocab[f])
            prior = np.bincount(y[tr & (y != IGNORE)], minlength=K).astype(float); prior /= prior.sum()
            mp = int(prior.argmax())
            last, mode = np.full(len(ev), mp), np.full(len(ev), mp)
            counts = np.zeros((len(ev), K))                 # frequenze nella storia della gang (ultimi 50)
            for _, g in ev.groupby("gang", sort=False):
                hist, prev = deque(maxlen=50), None
                for i in g.index:
                    if prev is not None: last[i] = prev
                    if hist:
                        c = Counter(hist); mode[i] = c.most_common(1)[0][0]
                        for k, v in c.items(): counts[i, k] = v
                    v = feat[f][i]
                    if v != unk[f] and not (f == "delta" and v == 0):
                        prev = v; hist.append(v)
            # GangMode probabilistico: frequenze della gang + alpha * prior globale;
            # alpha scelto sulla log-loss di validazione
            mval = (ev.split.values == "val") & (y != IGNORE)
            def gm_proba(a): P = counts + a * prior; return P / P.sum(1, keepdims=True)
            from sklearn.metrics import log_loss
            alpha = min((0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500), key=lambda a: log_loss(y[mval], np.clip(gm_proba(a)[mval], 1e-9, 1), labels=np.arange(K)))
            res["GangMode"]["alpha"][f] = alpha
            for split in ("val", "test"):
                m = ev.split.values == split
                res["MostPopular"][split][f] = metrics(y[m], np.full(m.sum(), mp), np.tile(prior, (m.sum(), 1)))
                res["LastValue"][split][f] = metrics(y[m], last[m])
                res["GangMode"][split][f] = metrics(y[m], mode[m], gm_proba(alpha)[m])
        for name, r in res.items():
            json.dump({"model": name, "version": version, "seed": 0, **r},
                      open(OUT / f"{name}_{version}_s0.json", "w"), indent=2)
            print(version, name, {f: round(r["test"][f]["weighted_f1"], 3) for f in FIELDS})


if __name__ == "__main__":
    main()
