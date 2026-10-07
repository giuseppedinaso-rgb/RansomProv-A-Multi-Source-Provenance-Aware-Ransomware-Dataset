#!/usr/bin/env python3
"""Rimappa i risultati per evento dopo una nuova esecuzione degli stadi 3-4.

Uso: python3 remap_events.py <tag_vecchio>
  legge  build/record_to_event_<tag>.json e build/events_<tag>.json (salvati
         prima di rieseguire gli stadi 3-4), e i nuovi build/record_to_event.json
         e build/events.json;
  scrive build/event_id_map_<tag>.json e riscrive ogni file di risultati
         (site_*.jsonl, llm_knowledge_*.jsonl), tenendo una copia .<tag>.
         Le copie storiche in build/archive/ non sono toccate.

Corrispondenza vecchio -> nuovo evento:
  1. tramite gli ID dei record grezzi (source, source_id): il nuovo evento che
     contiene la maggioranza dei record del vecchio. I record senza ID (alcuni
     record RansomLook) sono esclusi: un join su None li collegherebbe tutti;
  2. per i vecchi eventi senza record con ID: (gang, victim, discovered), solo
     se la combinazione e' univoca in entrambe le versioni e il nome ha almeno
     4 caratteri alfanumerici.
Un risultato che dipende dal dominio (site_*) e' tenuto solo se il nuovo evento
ha lo stesso dominio.
"""
import sys, json, re, glob, shutil, collections
from pathlib import Path
import pandas as pd

BUILD = Path("build")


def main(tag):
    rd = lambda f: pd.read_json(f, dtype={"source_id": str})
    o, n = rd(BUILD / f"record_to_event_{tag}.json"), rd(BUILD / "record_to_event.json")
    ok = lambda d: d[d.source_id.notna() & (d.source_id != "None")]
    m = ok(o).merge(ok(n), on=["source", "source_id"], suffixes=("_old", "_new"))
    mp = m.groupby("event_id_old").event_id_new.agg(
        lambda s: collections.Counter(s).most_common(1)[0][0]).astype(int)

    e1, e2 = pd.read_json(BUILD / f"events_{tag}.json"), pd.read_json(BUILD / "events.json")
    k = ["gang", "victim", "discovered"]
    good = lambda df: df[df.victim.map(lambda v: len(re.sub(r"[^a-z0-9]", "", str(v).lower())) >= 4)]
    u1 = good(e1[~e1.event_id.isin(mp.index)]); u1 = u1[~u1.duplicated(k, keep=False)]
    c2 = good(e2); c2 = c2[~c2.duplicated(k, keep=False)]
    fb = u1.merge(c2[k + ["event_id"]], on=k, suffixes=("", "_new"))
    mp = pd.concat([mp, fb.set_index("event_id").event_id_new.astype(int)])
    mp.to_json(BUILD / f"event_id_map_{tag}.json")
    print(f"eventi mappati {len(mp)} di {len(e1)} (fallback per nome {len(fb)})")

    dom = e2.set_index("event_id").domain
    files = sorted(set(glob.glob(str(BUILD / "site_*.jsonl")) + glob.glob(str(BUILD / "llm_knowledge_*.jsonl"))))
    for f in files:
        shutil.copy(f, f"{f}.{tag}")
        out, lost, changed = [], 0, 0
        for line in open(f"{f}.{tag}", encoding="utf-8"):
            r = json.loads(line)
            e = r["event_id"]
            if e not in mp.index:
                lost += 1; continue
            ne = int(mp[e])
            if "domain" in r and r.get("origin") != "invalid_domain" and dom.get(ne) != r["domain"]:
                changed += 1; continue
            r["event_id"] = ne
            out.append(r)
        with open(f, "w", encoding="utf-8") as fh:
            for r in out:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {Path(f).name}: {len(out)} tenuti | senza corrispondenza {lost} | dominio cambiato {changed}")


if __name__ == "__main__":
    main(sys.argv[1])
