#!/usr/bin/env python3
"""Stadio 6 — integra le risposte LLM accettate -> build/events_final.json.

Una risposta si accetta solo se:
  * l'attributo era davvero mancante nell'evento (need_country/need_sector);
  * non e' 'unknown' e non ha evidenza da un tracker ransomware;
  * la confidenza e' >= alla soglia fissata in llm_policy.json per quella
    modalita' (con/senza ricerca) e quell'attributo. Se la validazione non ha
    trovato una soglia che raggiunga l'obiettivo, quell'attributo in quella
    modalita' NON si applica affatto.

Provenienza: country_src / sector_src = 'llm_web' (con ricerca) o 'llm'
(senza), piu' llm_confidence, llm_evidence_url, llm_model, llm_prompt_version.
"""
import json
from pathlib import Path
from collections import Counter
import pandas as pd

BUILD = Path("build")


def main():
    pol = json.load(open("llm_policy.json", encoding="utf-8"))
    thr = pol["thresholds"]
    # parte dall'output dello stadio 5c (sito web) se presente
    src_file = BUILD / "events_site.json" if (BUILD / "events_site.json").exists() else BUILD / "events.json"
    ev = pd.read_json(src_file).set_index("event_id")
    for c in ("llm_confidence", "llm_evidence_url", "llm_model", "llm_prompt_version"):
        ev[c] = None

    stats = Counter()
    for line in open(BUILD / "llm_apply.jsonl", encoding="utf-8"):
        r = json.loads(line); p = r.get("parsed")
        if "error" in r or not p:
            stats["errore_o_non_parsabile"] += 1; continue
        if r.get("rejected") == "tracker_evidence":
            stats["scartata_tracker"] += 1; continue
        mode = "search" if r["search"] else "nosearch"
        src = "llm_web" if r["search"] else "llm"
        e = r["event_id"]
        applied = False
        for attr, val, need in (("country", p.get("country"), r.get("need_country")),
                                ("sector", p.get("sector"), r.get("need_sector"))):
            th = thr[f"{mode}_{attr}"]["min_confidence"]
            if not need or ev.at[e, f"{attr}_src"] != "missing":
                continue
            if val is None:
                stats[f"{attr}_unknown"] += 1; continue
            if th is None:
                stats[f"{attr}_{mode}_nessuna_soglia_valida"] += 1; continue
            if p["confidence"] < th:
                stats[f"{attr}_sotto_soglia"] += 1; continue
            ev.at[e, attr] = val
            ev.at[e, f"{attr}_src"] = src
            stats[f"{attr}_{src}"] += 1; applied = True
        if applied:
            ev.at[e, "llm_confidence"] = p["confidence"]
            ev.at[e, "llm_evidence_url"] = p.get("evidence_url")
            ev.at[e, "llm_model"] = r["model"]
            ev.at[e, "llm_prompt_version"] = r["prompt_version"]

    ev = ev.reset_index()
    ev.to_json(BUILD / "events_final.json", orient="records", force_ascii=False, indent=1)
    rep = {"applied": dict(stats), "country_src": ev.country_src.value_counts().to_dict(),
           "sector_src": ev.sector_src.value_counts().to_dict(), "policy": pol}
    json.dump(rep, open(BUILD / "report_stage6.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps({k: rep[k] for k in ("applied", "country_src", "sector_src")}, ensure_ascii=False, indent=2))
    print("-> build/events_final.json, build/report_stage6.json")


if __name__ == "__main__":
    main()
