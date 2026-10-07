#!/usr/bin/env python3
"""Rilascio — due versioni del dataset dallo stesso file master.

Input, il primo che esiste: build/events_final.json (stadio 6, LLM con ricerca),
build/events_knowledge.json (5e), build/events_site.json (5c), build/events.json (4).

  release/events_observed.{json,csv}
      Paese e settore SOLO se riportati dalle fonti per quell'evento
      (country_src / sector_src == 'observed'; i conflitti tra fonti sono
      risolti con le regole dello stadio 4, ma il valore resta uno di quelli
      osservati). Ogni altro valore e' null. Per chi vuole evitare qualsiasi
      inferenza.
  release/events_enriched.{json,csv}
      Tutti i valori, incluse le inferenze (eredita' dalla stessa vittima,
      ccTLD, descrizione, titolo, sito web, LLM). La colonna *_src dice da dove
      viene ogni valore, cosi' l'utente puo' filtrare le inferenze che non
      accetta.

Pulizia del campo domain (solo in uscita): le fonti riportano a volte in quel
campo testo che non e' un dominio (troncato 'foo.co...', indirizzi .onion,
frasi del leak site, 'u.s. gov'). Nel rilascio questi valori diventano null.
Gli stadi 3-4 restano invariati, cosi' gli event_id e tutti i file intermedi
(site_*.jsonl) restano allineati.
"""
import json, re
from ransomlook_clean import clean_victim
from pathlib import Path
import pandas as pd

import os
BUILD, OUT = Path("build"), Path(os.environ.get("RELEASE_DIR", "release"))
DOMAIN_OK = re.compile(r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}")

COLS = ["event_id", "discovered", "victim", "victim_raw", "victim_truncated", "victim_placeholder", "not_a_victim", "domain", "gang", "gang_raw",
        "country", "country_src", "country_resolution", "country_conflict", "country_candidates",
        "sector", "sector_src", "sector_resolution", "sector_conflict", "sector_candidates",
        "sources", "n_sources"]
ENRICH_COLS = ["site_rule", "site_snapshot", "sector_model", "sector_confidence",
               "knowledge_input", "knowledge_model", "knowledge_confidence_country", "knowledge_confidence_sector",
               "llm_confidence", "llm_evidence_url", "llm_model", "llm_prompt_version"]


# titoli che non nominano una vittima: segnaposto dei leak site ("FILES",
# "ARCHIVE6", "Views: 23", "Company", "(SOLD TO 3RD PARTY)") o titoli troncati
# a soli puntini. Gli eventi restano nel dataset (l'attacco e' stato
# rivendicato) ma sono segnalati in victim_placeholder.
PLACEHOLDER = re.compile(
    r"^\s*(?:views:\s*\d+|archive\s*\d*|files?|company|protected|full data ?dump(?: free)?|breach|"
    r"disclose|auction|sold|\(?sold to 3rd party\)?|announcement[\s.:\d/-]*|test|unknown|n/?a|"
    r"redacted|\.+|-+)\s*$", re.I)


def clean_domain(d):
    d = (d or "").strip().lower()
    return d if DOMAIN_OK.fullmatch(d) and not d.endswith(".onion") else None


def save(df, name):
    df.to_json(OUT / f"{name}.json", orient="records", force_ascii=False, indent=1)
    flat = df.copy()
    for c in flat.columns:
        if flat[c].map(lambda v: isinstance(v, list)).any():
            flat[c] = flat[c].map(lambda v: "|".join(map(str, v)) if isinstance(v, list) else v)
    flat.to_csv(OUT / f"{name}.csv", index=False)


def summary(df):
    return {"events": len(df), "victims": int(df.victim.nunique()), "gangs": int(df.gang.nunique()),
            "countries": int(df.country.nunique()), "sectors": int(df.sector.nunique()),
            "first": str(df.discovered.min()), "last": str(df.discovered.max()),
            "country_missing_pct": round(df.country.isna().mean() * 100, 2),
            "sector_missing_pct": round(df.sector.isna().mean() * 100, 2),
            "country_src": df.country_src.value_counts().to_dict(),
            "sector_src": df.sector_src.value_counts().to_dict()}


def main():
    src = next(p for p in (BUILD / "events_final.json", BUILD / "events_knowledge.json",
                           BUILD / "events_site.json", BUILD / "events.json")
               if p.exists())
    ev = pd.read_json(src, dtype={"site_snapshot": str})
    ev["discovered"] = pd.to_datetime(ev.discovered).dt.strftime("%Y-%m-%d")
    ev["victim_raw"] = ev.victim
    ev["victim"] = ev.victim.map(clean_victim)
    # nomi troncati dal leak site ("acme-industr..."): non ricostruibili, segnalati
    # v1.1: post che non nominano una vittima ma un forum, la gang stessa, un
    # comunicato o un leak di dati di molte organizzazioni
    gang_keys = set(ev.gang.astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True))
    vk = ev.victim.astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    ev["not_a_victim"] = (vk.isin(gang_keys) & (vk.str.len() >= 4)) | ev.victim.astype(str).str.contains(
        r"(?i)breachforums|^\W*(?:press release|(?:first |second |final )?statement|announcement)[\W\d]*$|^admins?$|fortigate",
        regex=True)
    ev["victim_placeholder"] = ev.victim.astype(str).str.match(PLACEHOLDER) & ev.domain.isna()
    ev["victim_truncated"] = ev.victim_raw.astype(str).str.contains(r"(?:\.\.\.|…)\s*$", regex=True)
    bad = ev.domain.notna() & ev.domain.map(clean_domain).isna()
    ev["domain"] = ev.domain.map(clean_domain)
    for c in ("country", "sector"):
        ev.loc[ev[f"{c}_src"] == "missing", c] = None
    OUT.mkdir(exist_ok=True)

    enriched = ev[COLS + [c for c in ENRICH_COLS if c in ev.columns]].copy()
    if "site_snapshot" in enriched:
        enriched["site_snapshot"] = enriched.site_snapshot.map(
            lambda v: None if v in (None, "None", "nan") or pd.isna(v) else str(v).split(".")[0])
    save(enriched, "events_enriched")

    observed = ev[COLS].copy()
    for c in ("country", "sector"):
        inferred = observed[f"{c}_src"] != "observed"
        observed.loc[inferred, c] = None
        observed.loc[inferred, f"{c}_src"] = "missing"
    save(observed, "events_observed")

    rep = {"source_file": str(src), "domains_set_to_null": int(bad.sum()),
           "events_observed": summary(observed), "events_enriched": summary(enriched)}
    json.dump(rep, open(OUT / "release_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
