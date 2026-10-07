#!/usr/bin/env python3
"""Test 1 — una sola fonte contro il dataset unito.

Domanda: perche' armonizzare tre tracker invece di usarne uno? Per ogni fonte
(ransomware.live, ransomfeed, RansomLook) si costruisce la vista a fonte
singola: eventi visti da quella fonte, con paese e settore riportati solo da
quella fonte (nextevent_data.load(..., source)). Si confronta con il dataset unito.

A. Copertura: eventi, gang, anni coperti, completezza di paese e settore.
B. Stabilita' delle conclusioni descrittive (2021-2026), rispetto al dataset unito:
   correlazione dei conteggi settimanali; Spearman della classifica delle 20
   gang piu' attive; quota USA; divergenza di Jensen-Shannon delle
   distribuzioni di paese (top 10 + altri) e settore; ultima rivendicazione
   prima delle operazioni di polizia.
C. Previsione del prossimo evento sugli STESSI eventi di test (quelli visti
   dalla fonte): GangMode con storia della sola fonte contro storia unita (CPU);
   GRU ottimizzato idem (GPU, single_source_gru.py).
Uscita: benchmark/results/single_source.json
"""
import json
from pathlib import Path
from collections import Counter, deque
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon
from nextevent_data import load, targets, metrics, FIELDS, IGNORE, ROOT

SOURCES = ["ransomlive", "ransomfeed", "ransomlook"]
TAKEDOWNS = {"hive": "2023-01-26", "ragnarlocker": "2023-10-19", "lockbit3": "2024-02-19", "8base": "2025-02-10",
             "ransomhub": "2025-04-01", "dispossessor": "2024-08-12"}


def dist(s, top):
    v = s.dropna(); v = v.where(v.isin(top), "other")
    return v.value_counts(normalize=True).reindex(list(top) + ["other"], fill_value=0).values


def gangmode(ev, vocab, tgt, f, alpha=50):
    feat = {"delta": ev.delta.values, "country": ev.country_i.values, "sector": ev.sector_i.values}[f]
    K = len(vocab[f]); unk = {"delta": 0, "country": K - 1, "sector": K - 1}[f]
    tr = ev.split.values == "train"; y = tgt[f]
    prior = np.bincount(y[tr & (y != IGNORE)], minlength=K).astype(float) + 1e-6; prior /= prior.sum()
    counts = np.zeros((len(ev), K))
    for _, g in ev.groupby("gang", sort=False):
        hist = deque(maxlen=50)
        for i in g.index:
            for k, c in Counter(hist).items(): counts[i, k] = c
            v = feat[i]
            if v != unk and not (f == "delta" and v == 0): hist.append(v)
    P = counts + (5 if f == "delta" else alpha) * prior
    return P / P.sum(1, keepdims=True)


def main():
    rep = {"coverage": {}, "conclusions": {}, "next_event_gangmode": {}}
    merged = pd.read_json(ROOT / "release" / "events_observed.json"); merged["d"] = pd.to_datetime(merged.discovered)
    views = {"merged": merged.assign(country_v=merged.country, sector_v=merged.sector)}
    from nextevent_data import source_attributes
    for s in SOURCES:
        v = merged[merged.sources.map(lambda x: s in x)].copy(); sa = source_attributes(s)
        v["country_v"] = v.event_id.map(sa.country_src1); v["sector_v"] = v.event_id.map(sa.sector_src1)
        views[s] = v
    W = pd.date_range("2021-01-04", "2026-09-28", freq="W-MON")
    wk = lambda v: v[v.d >= "2021-01-01"].groupby(v.d.dt.to_period("W-SUN").dt.start_time).size().reindex(W, fill_value=0)
    m21 = merged[merged.d >= "2021-01-01"]
    top_g = m21.gang.value_counts().head(20).index
    top_c = merged.country.value_counts().head(10).index
    secs = sorted(merged.sector.dropna().unique())
    for name, v in views.items():
        rep["coverage"][name] = {"events": len(v), "share_of_merged": round(len(v) / len(merged) * 100, 1),
                                 "gangs": int(v.gang.nunique()), "first": str(v.d.min().date()),
                                 "country_pct": round(v.country_v.notna().mean() * 100, 1),
                                 "sector_pct": round(v.sector_v.notna().mean() * 100, 1),
                                 "events_by_year": v.d.dt.year.value_counts().sort_index().to_dict()}
        if name == "merged":
            continue
        v21 = v[v.d >= "2021-01-01"]
        gc = v21.gang.value_counts().reindex(top_g, fill_value=0)
        con = {"weekly_corr_2021_26": round(np.corrcoef(wk(v), wk(merged))[0, 1], 3),
               "weekly_corr_2023_26": round(np.corrcoef(wk(v)[W >= "2023-01-01"], wk(merged)[W >= "2023-01-01"])[0, 1], 3),
               "top20_gang_spearman": round(spearmanr(gc.values, m21.gang.value_counts().reindex(top_g).values)[0], 3),
               "top20_gangs_missing": int((gc == 0).sum())}
        if v.country_v.notna().any():
            con["us_share_pct"] = round((v.country_v == "US").sum() / v.country_v.notna().sum() * 100, 1)
            con["country_JS"] = round(float(jensenshannon(dist(v.country_v, top_c), dist(merged.country, top_c))), 3)
        if v.sector_v.notna().any():
            con["sector_JS"] = round(float(jensenshannon(dist(v.sector_v, secs), dist(merged.sector, secs))), 3)
        con["takedowns_last_claim_days_before"] = {
            g: (int((pd.Timestamp(d) - v[(v.gang == g) & (v.d < d)].d.max()).days) if ((v.gang == g) & (v.d < d)).any() else None)
            for g, d in TAKEDOWNS.items()}
        rep["conclusions"][name] = con
    rep["conclusions"]["merged_us_share_pct"] = round((merged.country == "US").sum() / merged.country.notna().sum() * 100, 1)

    # C. prossimo evento, GangMode: stessi eventi di test (quelli visti dalla fonte)
    evm, vocab = load("observed"); tm = targets(evm, vocab)
    Pm = {f: gangmode(evm, vocab, tm, f) for f in FIELDS}
    rowm = pd.Series(range(len(evm)), index=evm.event_id)
    for s in SOURCES:
        evs, _ = load("observed", s); ts = targets(evs, vocab)
        test = evs.split.values == "test"; ids = evs.event_id.values[test]
        out = {}
        for f in FIELDS:
            Ps = gangmode(evs, vocab, ts, f)[test]; y = ts[f][test]
            Pmm = Pm[f][rowm.loc[ids].values]
            out[f] = {"single_source": metrics(y, Ps.argmax(1), Ps), "merged_history": metrics(y, Pmm.argmax(1), Pmm)}
        rep["next_event_gangmode"][s] = out
    json.dump(rep, open(ROOT / "benchmark" / "results" / "single_source.json", "w"), indent=1, default=str)
    print(json.dumps({k: rep[k] for k in ("coverage", "conclusions")}, indent=1, default=str))
    for s, o in rep["next_event_gangmode"].items():
        print(s, {f: (round(o[f]["single_source"]["log_loss"], 3), round(o[f]["merged_history"]["log_loss"], 3),
                      round(o[f]["single_source"]["accuracy"], 3), round(o[f]["merged_history"]["accuracy"], 3)) for f in FIELDS})


if __name__ == "__main__":
    main()
