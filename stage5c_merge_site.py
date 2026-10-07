#!/usr/bin/env python3
"""Stadio 5c — integra i paesi ricavati dal sito della vittima -> build/events_site.json.

Input: build/events.json (stadio 4) e build/site_apply.jsonl (site_enrich.py apply).

Una risposta si accetta solo se:
  * il paese dell'evento era davvero mancante (country_src == 'missing');
  * la regola di decisione e' tra quelle validate (ACCEPT_RULES). Le precisioni
    misurate su 600 eventi a paese noto (site_validate.jsonl) sono:
    strong 96.3% (n=218), phone 94.6% (n=92), phone_majority 2/2;
    per origine archive 96.0% (n=272), live 94.3% (n=35).

Provenienza: country_src = 'site_archive' se la pagina viene da uno snapshot
di archive.org precedente all'attacco, 'site_live' se viene dal sito attuale
(nessuno snapshot disponibile). Colonne aggiunte: site_rule, site_snapshot,
site_signals.

Settore (site_sector.py apply): si accetta se il settore era mancante, la
risposta non e' 'unknown' e la confidenza e' >= SECTOR_MIN_CONF. Soglia scelta
sulla validazione di gpt-oss-120b (300 eventi): a 0.90 accuratezza 80.8% sugli
eventi con fonti concordi, copertura 78%. Rieseguita interamente in locale
(vllm, stessi pesi pubblici): 82.3%, accordo con Groq 97.3%. Se esiste
site_sector_apply_local.jsonl si usa solo quello (un solo modello e backend
per tutto il dataset); il modello e' registrato per evento in sector_model.
sector_src = 'site_llm'.
"""
import json
from pathlib import Path
from collections import Counter
import pandas as pd

BUILD = Path("build")
ACCEPT_RULES = {"strong", "phone", "phone_majority"}
SECTOR_MIN_CONF = 0.90


def main():
    ev = pd.read_json(BUILD / "events.json").set_index("event_id")
    for c in ("site_rule", "site_snapshot", "site_signals"):
        ev[c] = None

    stats = Counter()
    for line in open(BUILD / "site_apply.jsonl", encoding="utf-8"):
        r = json.loads(line)
        e = r["event_id"]
        if e not in ev.index:
            stats["evento_assente"] += 1; continue
        if ev.at[e, "country_src"] != "missing":
            stats["paese_gia_noto"] += 1; continue
        if not r.get("fetched"):
            stats[f"non_recuperata:{r.get('origin', 'errore')}"] += 1; continue
        if not r.get("country"):
            stats[f"nessuna_risposta:{r.get('rule')}"] += 1; continue
        if r["rule"] not in ACCEPT_RULES:
            stats[f"regola_non_accettata:{r['rule']}"] += 1; continue
        origin = r["origin"]
        src = "site_archive" if origin.startswith("archive:") else "site_live"
        ev.at[e, "country"] = r["country"]
        ev.at[e, "country_src"] = src
        ev.at[e, "site_rule"] = r["rule"]
        ev.at[e, "site_snapshot"] = origin[8:] if origin.startswith("archive:") else None
        ev.at[e, "site_signals"] = r.get("signals")
        stats[f"applicata:{src}"] += 1

    ev["sector_model"] = None
    ev["sector_confidence"] = None
    # se esiste la corsa locale (gpt-oss-120b via vllm, stesso modello per
    # validazione e applicazione) si usa solo quella; altrimenti i file Groq
    local = BUILD / "site_sector_apply_local.jsonl"
    sector_files = [local] if local.exists() else sorted(BUILD.glob("site_sector_apply*.jsonl"))
    for fp in sector_files:
        for line in open(fp, encoding="utf-8"):
            r = json.loads(line)
            e, p = r["event_id"], r.get("parsed") or {}
            if "error" in r or e not in ev.index:
                stats["settore_errore"] += 1; continue
            if ev.at[e, "sector_src"] != "missing":
                stats["settore_gia_noto"] += 1; continue
            if not p.get("sector"):
                stats["settore_unknown"] += 1; continue
            if p["confidence"] < SECTOR_MIN_CONF:
                stats["settore_sotto_soglia"] += 1; continue
            ev.at[e, "sector"] = p["sector"]
            ev.at[e, "sector_src"] = "site_llm"
            ev.at[e, "sector_model"] = r["model"] + (" (vllm local)" if r.get("backend") == "vllm-local" else " (groq)")
            ev.at[e, "sector_confidence"] = p["confidence"]
            stats[f"settore_applicato:{ev.at[e, 'sector_model']}"] += 1

    ev = ev.reset_index()
    ev.to_json(BUILD / "events_site.json", orient="records", force_ascii=False, indent=1)
    rep = {"stats": dict(sorted(stats.items())),
           "country_src": ev.country_src.value_counts().to_dict(),
           "country_missing_pct": round((ev.country_src == "missing").mean() * 100, 2),
           "sector_src": ev.sector_src.value_counts().to_dict(),
           "sector_missing_pct": round((ev.sector_src == "missing").mean() * 100, 2),
           "accepted_rules": sorted(ACCEPT_RULES), "sector_min_conf": SECTOR_MIN_CONF}
    json.dump(rep, open(BUILD / "report_stage5c.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    print("-> build/events_site.json, build/report_stage5c.json")


if __name__ == "__main__":
    main()
