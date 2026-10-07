#!/usr/bin/env python3
"""Benchmark 3 — dati per i processi puntuali marcati (EasyTPP).

Sequenza = rivendicazioni di una gang in un periodo; tipo di evento (mark) =
settore (13 classi + 'unknown') oppure paese (9 piu' frequenti + 'other' +
'unknown'). La classe 'unknown' tiene nella sequenza gli eventi senza valore,
cosi' i tempi non vengono alterati; in observed e' piu' frequente che in
enriched.

Divisione temporale senza fughe di informazione: ogni gang contribuisce con
sequenze separate per periodo
  train  2021-01-01 .. 2024-12-31
  dev    2025-01-01 .. 2025-06-30
  test   2025-07-01 .. 2026-09-30
Il tempo riparte da 0 all'inizio di ogni sequenza (giorni). Sequenze lunghe sono
spezzate in blocchi consecutivi di al massimo MAX_LEN eventi; si tengono solo
quelle con almeno MIN_LEN eventi.

Le date hanno risoluzione giornaliera e molte gang pubblicano piu' vittime lo
stesso giorno: gli eventi dello stesso giorno sono distribuiti uniformemente
dentro la giornata (i / n), in ordine di event_id, per evitare intervalli nulli.

Uscita: benchmark/tpp_data/<version>_<mark>/{train,dev,test}.pkl
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark" / "tpp_data"
PERIODS = {"train": ("2021-01-01", "2024-12-31"), "dev": ("2025-01-01", "2025-06-30"),
           "test": ("2025-07-01", "2026-09-30")}
MAX_LEN, MIN_LEN = 200, 3


def marks(ev, mark, top_countries):
    if mark == "sector":
        vocab = sorted(pd.read_json(ROOT / "release" / "events_observed.json").sector.dropna().unique()) + ["unknown"]
        lab = ev.sector.fillna("unknown")
    else:
        vocab = list(top_countries) + ["other", "unknown"]
        lab = ev.country.map(lambda c: "unknown" if pd.isna(c) else (c if c in top_countries else "other"))
    return lab.map({v: i for i, v in enumerate(vocab)}), vocab


def to_events(t, k):
    evs, last_same = [], {}
    for i, (ti, ki) in enumerate(zip(t, k)):
        dt = 0.0 if i == 0 else ti - t[i - 1]
        evs.append({"idx_event": i, "time_since_start": float(ti), "time_since_last_event": float(dt),
                    "time_since_last_same_event": float(ti - last_same.get(ki, t[0])), "type_event": int(ki)})
        last_same[ki] = ti
    return evs


def build(version, mark):
    ev = pd.read_json(ROOT / "release" / f"events_{version}.json")
    ev["d"] = pd.to_datetime(ev.discovered)
    obs = pd.read_json(ROOT / "release" / "events_observed.json")
    obs["d"] = pd.to_datetime(obs.discovered)
    top = list(obs[obs.d <= "2024-12-31"].country.value_counts().head(9).index)
    ev["k"], vocab = marks(ev, mark, top)
    ev = ev.sort_values(["d", "event_id"])
    # ora dentro la giornata
    ev["n_day"] = ev.groupby(["gang", "d"]).event_id.transform("size")
    ev["i_day"] = ev.groupby(["gang", "d"]).cumcount()
    ev["t_abs"] = (ev.d - pd.Timestamp("2021-01-01")).dt.days + ev.i_day / ev.n_day
    data, stats = {}, {}
    for split, (a, b) in PERIODS.items():
        seqs = []
        sub = ev[(ev.d >= a) & (ev.d <= b)]
        for _, g in sub.groupby("gang"):
            t, k = g.t_abs.values, g.k.values
            for s in range(0, len(t), MAX_LEN):
                tt, kk = t[s:s + MAX_LEN], k[s:s + MAX_LEN]
                if len(tt) >= MIN_LEN:
                    seqs.append(to_events(tt - tt[0], kk))
        data[split] = seqs
        n = sum(len(s) for s in seqs)
        unk = sum(e["type_event"] == len(vocab) - 1 for s in seqs for e in s)
        stats[split] = {"sequences": len(seqs), "events": n, "unknown_pct": round(unk / max(n, 1) * 100, 1)}
    d = OUT / f"{version}_{mark}"
    d.mkdir(parents=True, exist_ok=True)
    for split, seqs in data.items():
        pickle.dump({"dim_process": len(vocab), split: seqs}, open(d / f"{split}.pkl", "wb"))
    (d / "vocab.txt").write_text("\n".join(vocab))
    return len(vocab), stats


def main():
    for version in ("observed", "enriched"):
        for mark in ("sector", "country"):
            k, st = build(version, mark)
            print(version, mark, "tipi", k, st)


if __name__ == "__main__":
    main()
