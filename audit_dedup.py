#!/usr/bin/env python3
"""Audit della deduplicazione -> build/audit_dedup.json e audit/ (campioni da etichettare).

Due tipi di errore:
  fusione sbagliata   un evento contiene record di vittime diverse.
                      Candidati: eventi i cui record hanno nomi poco simili
                      (similarita' minima < 0.5 tra i nomi puliti).
  duplicato perso     lo stesso attacco e' diviso in due eventi.
                      Candidati: coppie di eventi della stessa gang entro 7
                      giorni con nomi simili (>= 0.85) o stesso dominio.

I candidati sono un limite superiore: molti sono corretti (stessa vittima con
titoli diversi; vittime diverse con nomi simili). Lo script estrae un campione
casuale di ciascun tipo in audit/*.csv con una colonna 'label' vuota, da
compilare a mano (ok / errore / dubbio), per stimare il tasso di errore reale.
"""
import re, json
from pathlib import Path
from difflib import SequenceMatcher
import pandas as pd
from ransomlook_clean import clean_victim

BUILD, AUDIT = Path("build"), Path("audit")
N_SAMPLE = 100
norm = lambda s: re.sub(r"[^a-z0-9]", "", clean_victim(s).lower())


def main():
    AUDIT.mkdir(exist_ok=True)
    r = pd.read_json(BUILD / "record_to_event.json", dtype={"source_id": str}).drop_duplicates(["source", "source_id"])
    u = pd.read_json(BUILD / "unified.json", dtype={"source_id": str}).merge(r, on=["source", "source_id"])
    u["nk"] = u.victim.map(norm)

    def minsim(ks):
        ks = [k for k in set(ks) if k]
        return min((SequenceMatcher(None, a, b).ratio() for i, a in enumerate(ks) for b in ks[i + 1:]), default=1.0)
    ms = u.groupby("event_id").nk.agg(minsim)
    merge_c = ms[ms < 0.5]
    agg = u[u.event_id.isin(merge_c.index)].groupby("event_id").agg(
        gang=("gang", "first"), names=("victim", lambda s: " || ".join(list(dict.fromkeys(s))[:5])),
        domains=("domain", lambda s: " || ".join(list(dict.fromkeys(s.dropna()))[:4])), n_records=("source", "size"))
    agg.sample(min(N_SAMPLE, len(agg)), random_state=0).assign(label="").to_csv(AUDIT / "merge_candidates_sample.csv")

    ev = pd.read_json(BUILD / "events.json")
    ev["nk"], ev["d"] = ev.victim.map(norm), pd.to_datetime(ev.discovered)
    pairs = []
    for g, grp in ev.groupby("gang"):
        rows = grp.sort_values("d")[["event_id", "nk", "d", "domain", "victim"]].values
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                gap = (rows[j][2] - rows[i][2]).days
                if gap > 7:
                    break
                a, b = rows[i][1], rows[j][1]
                if not a or not b:
                    continue
                s = SequenceMatcher(None, a, b).ratio()
                if s >= 0.85 or (rows[i][3] and rows[i][3] == rows[j][3]):
                    pairs.append(dict(gang=g, event_1=rows[i][0], event_2=rows[j][0], victim_1=rows[i][4],
                                      victim_2=rows[j][4], domain_1=rows[i][3], domain_2=rows[j][3],
                                      gap_days=gap, similarity=round(s, 2)))
    P = pd.DataFrame(pairs)
    P.sample(min(N_SAMPLE, len(P)), random_state=0).assign(label="").to_csv(AUDIT / "duplicate_candidates_sample.csv", index=False)

    rep = {"events": len(ev), "events_multi_name": int((ms < 1).sum()),
           "merge_candidates": int(len(merge_c)), "duplicate_candidate_pairs": int(len(P)),
           "duplicate_pairs_within_3_days": int((P.gap_days <= 3).sum()) if len(P) else 0,
           "largest_events_records": r.event_id.value_counts().head(5).tolist()}
    json.dump(rep, open(BUILD / "audit_dedup.json", "w"), indent=2)
    print(json.dumps(rep, indent=2))
    print("-> audit/merge_candidates_sample.csv, audit/duplicate_candidates_sample.csv")


if __name__ == "__main__":
    main()
