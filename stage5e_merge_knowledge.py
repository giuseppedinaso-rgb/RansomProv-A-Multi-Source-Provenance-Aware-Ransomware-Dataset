#!/usr/bin/env python3
"""Stadio 5e — integra le risposte di llm_knowledge.py -> build/events_knowledge.json.

Input: build/events_site.json (stadio 5c) e build/llm_knowledge_apply.jsonl.

Una risposta si accetta solo se:
  * l'attributo e' ancora mancante nell'evento;
  * la risposta non e' 'unknown';
  * per quel (attributo, tipo di input) llm_knowledge_policy.json ha una soglia
    e la confidenza la raggiunge. Le soglie vengono dalla validazione su eventi
    scoperti DOPO il cutoff di addestramento del modello (luglio 2024): prima di
    quella data il modello potrebbe ricordare le pagine dei tracker ransomware,
    cioe' le stesse fonti, e la precisione misurata sarebbe gonfiata. Con il
    solo nome della vittima non c'e' abbastanza evidenza non contaminata:
    quel tipo di input non si applica.

Provenienza: country_src / sector_src = 'llm_knowledge', piu'
knowledge_input (tipo di input), knowledge_model, knowledge_confidence_*.
"""
import json
from pathlib import Path
from collections import Counter
import pandas as pd

BUILD = Path("build")


def main():
    pol = json.load(open("llm_knowledge_policy.json", encoding="utf-8"))["rules"]
    ev = pd.read_json(BUILD / "events_site.json", dtype={"site_snapshot": str}).set_index("event_id")
    for c in ("knowledge_input", "knowledge_model", "knowledge_confidence_country", "knowledge_confidence_sector"):
        ev[c] = None

    stats = Counter()
    for line in open(BUILD / "llm_knowledge_apply.jsonl", encoding="utf-8"):
        r = json.loads(line)
        e, p = r["event_id"], r.get("parsed")
        if "error" in r or not p or e not in ev.index:
            stats["errore_o_assente"] += 1; continue
        applied = False
        for attr in ("country", "sector"):
            if ev.at[e, f"{attr}_src"] != "missing":
                continue
            rule = pol.get(f"{attr}|{r['itype']}")
            if not p.get(attr):
                stats[f"{attr}:unknown"] += 1; continue
            if rule is None:
                stats[f"{attr}:{r['itype']}:non_applicato"] += 1; continue
            if p[f"{attr}_confidence"] < rule["min_confidence"]:
                stats[f"{attr}:{r['itype']}:sotto_soglia"] += 1; continue
            ev.at[e, attr] = p[attr]
            ev.at[e, f"{attr}_src"] = "llm_knowledge"
            ev.at[e, f"knowledge_confidence_{attr}"] = p[f"{attr}_confidence"]
            stats[f"{attr}:{r['itype']}:applicato"] += 1
            applied = True
        if applied:
            ev.at[e, "knowledge_input"] = r["itype"]
            ev.at[e, "knowledge_model"] = f"{r['model']} (vllm local, {r['prompt_version']})"

    ev = ev.reset_index()
    ev.to_json(BUILD / "events_knowledge.json", orient="records", force_ascii=False, indent=1)
    rep = {"stats": dict(sorted(stats.items())), "policy": pol,
           "country_src": ev.country_src.value_counts().to_dict(),
           "country_missing_pct": round((ev.country_src == "missing").mean() * 100, 2),
           "sector_src": ev.sector_src.value_counts().to_dict(),
           "sector_missing_pct": round((ev.sector_src == "missing").mean() * 100, 2)}
    json.dump(rep, open(BUILD / "report_stage5e.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps({k: rep[k] for k in rep if k != "policy"}, ensure_ascii=False, indent=2))
    print("-> build/events_knowledge.json, build/report_stage5e.json")


if __name__ == "__main__":
    main()
