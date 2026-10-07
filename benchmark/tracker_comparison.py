#!/usr/bin/env python3
"""Punto 4 — confronto tra tracker con le metriche di Li et al.,
"Reading the Tea Leaves: A Comparative Analysis of Threat Intelligence",
USENIX Security 2019, adattate ai leak site.

Per ogni tracker (ransomware.live = RL, ransomfeed = RF, RansomLook = LK), sul
dataset v1.0 (release/) e per anno:
  volume           eventi visti dal tracker
  esclusivita'     quota dei suoi eventi che nessun altro tracker vede, e
                   contributo esclusivo al totale (Li et al.: "exclusive
                   contribution")
  differenziazione per coppia, quota degli eventi di A che B non ha
  intersezione     coefficiente di Jaccard tra coppie
  tempestivita'    ritardo dalla prima comparsa in qualsiasi tracker (giorni):
                   mediana, quota stesso giorno, quota in cui e' il primo
  copertura dei gruppi   gang viste / gang totali
  accuratezza      contro le vittime ransomware CONFERMATE (Washington AG):
                   quota delle vittime confermate presenti nel dataset che il
                   tracker vede (recall), e accordo del settore del tracker con
                   la categoria ufficiale (sugli abbinamenti)
  completezza      quota di eventi con paese / settore dalla fonte stessa
Uscita: benchmark/results/tracker_comparison.json
"""
import json
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = {"ransomlive": "RL", "ransomfeed": "RF", "ransomlook": "LK"}
MAP = {"Healthcare": "Health", "Financial Services": "Finance", "Education": "Education",
       "Government & Defense": "Government", "Other": "Non-Profit/Charity"}


def main():
    ev = pd.read_json(ROOT / "release" / "events_observed.json")
    ev["d"] = pd.to_datetime(ev.discovered); ev["year"] = ev.d.dt.year
    sets = {s: set(ev.event_id[ev.sources.map(lambda x: s in x)]) for s in SRC}
    r = pd.read_json(ROOT / "build" / "record_to_event_v1.0.json", dtype={"source_id": str})
    u = pd.read_json(ROOT / "build" / "unified.json", dtype={"source_id": str}, convert_dates=["discovered"])
    # unified.json e' ora v1.1: si usano solo i record con ID presenti nella mappa v1.0
    rec = u.merge(r[r.source_id.notna() & (r.source_id != "None")].drop_duplicates(["source", "source_id"]),
                  on=["source", "source_id"])
    first = rec.groupby(["event_id", "source"]).discovered.min().unstack()
    first_any = first.min(axis=1)
    rep = {"overall": {}, "pairs": {}, "by_year": {}}
    allg = ev.gang.nunique()
    for s, short in SRC.items():
        mine = ev[ev.event_id.isin(sets[s])]
        others = set().union(*[sets[o] for o in SRC if o != s])
        excl = sets[s] - others
        lag = (first[s] - first_any).dt.days.dropna() if s in first else pd.Series(dtype=float)
        attrs = rec[rec.source == s].groupby("event_id").agg(c=("country", lambda x: x.notna().any()),
                                                             k=("sector", lambda x: x.notna().any()))
        rep["overall"][short] = {
            "volume": len(sets[s]), "share_of_union_pct": round(len(sets[s]) / len(ev) * 100, 1),
            "exclusive_events": len(excl), "exclusive_pct_of_own": round(len(excl) / len(sets[s]) * 100, 1),
            "exclusive_contribution_pct_of_union": round(len(excl) / len(ev) * 100, 1),
            "gangs_seen": int(mine.gang.nunique()), "gang_coverage_pct": round(mine.gang.nunique() / allg * 100, 1),
            "latency_median_days": float(lag.median()), "latency_p90_days": float(lag.quantile(.9)),
            "same_day_as_first_pct": round((lag == 0).mean() * 100, 1),
            "own_country_pct": round(attrs.c.mean() * 100, 1), "own_sector_pct": round(attrs.k.mean() * 100, 1)}
        rep["by_year"][short] = {int(y): {"volume": int(g.event_id.isin(sets[s]).sum()),
                                          "exclusive": int(g.event_id.isin(excl).sum())}
                                 for y, g in ev[ev.year >= 2020].groupby("year")}
    for a, b in combinations(SRC, 2):
        A, B = sets[a], sets[b]
        rep["pairs"][f"{SRC[a]}-{SRC[b]}"] = {"jaccard": round(len(A & B) / len(A | B), 3),
                                               f"{SRC[a]}_not_in_{SRC[b]}_pct": round(len(A - B) / len(A) * 100, 1),
                                               f"{SRC[b]}_not_in_{SRC[a]}_pct": round(len(B - A) / len(B) * 100, 1)}
    # accuratezza contro le vittime confermate (Washington AG)
    wa = pd.read_csv(ROOT / "audit" / "wa_matches.csv")
    wa = wa[(wa.kind == "ransomware") & (wa.review_assistant == "correct")]
    raw = pd.read_csv(ROOT / "build" / "ext_wa_raw.csv")
    ind = raw.drop_duplicates("name").set_index("name").industrytype
    m = wa.merge(ev[["event_id", "sources"]], on="event_id", how="left")
    srec = rec[rec.event_id.isin(m.event_id)].groupby(["event_id", "source"]).sector.agg(
        lambda x: x.dropna().mode().iloc[0] if x.notna().any() else None).unstack()
    rep["confirmed_victims"] = {"n_confirmed_in_dataset": len(m)}
    for s, short in SRC.items():
        seen = m.sources.map(lambda x: isinstance(x, list) and s in x)
        out = {"recall_pct": round(seen.mean() * 100, 1)}
        if s in srec:
            sec = m.event_id.map(srec[s]).dropna()
            if len(sec):
                pred = sec.map(lambda v: MAP.get(v, "Business"))
                truth = m.loc[sec.index, "name"].map(ind)
                out["sector_agreement_pct"] = round((pred.values == truth.values).mean() * 100, 1)
                out["sector_n"] = int(len(sec))
        rep["confirmed_victims"][short] = out
    json.dump(rep, open(ROOT / "benchmark" / "results" / "tracker_comparison.json", "w"), indent=1)
    print(json.dumps(rep["overall"], indent=1)); print(json.dumps(rep["pairs"], indent=1))
    print(json.dumps(rep["confirmed_victims"], indent=1))
    print(pd.DataFrame({k: {y: v["exclusive"] for y, v in d.items()} for k, d in rep["by_year"].items()}).to_string())


if __name__ == "__main__":
    main()
