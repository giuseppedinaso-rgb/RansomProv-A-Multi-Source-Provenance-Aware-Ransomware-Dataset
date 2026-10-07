#!/usr/bin/env python3
"""Stadio 5e — paese e settore dalla conoscenza del modello (senza ricerca web).

Per gli eventi che restano senza paese o settore dopo gli stadi 4, 5c e 5d,
quasi sempre senza dominio, si chiede a gpt-oss-120b (in locale con vllm,
stessi pesi pubblici) di identificare l'organizzazione dal NOME, dal dominio se
c'e', e dalla descrizione del post sul leak site se c'e'.

Nessuna ricerca web: il modello usa solo cio' che sa. Per questo:
  * 'unknown' e' esplicitamente ammesso e preferito a una risposta incerta;
  * la validazione e' stratificata per tipo di input (solo nome / nome+dominio /
    nome+descrizione), perche' la qualita' attesa e' molto diversa;
  * la soglia di confidenza e' scelta per tipo di input e attributo
    (llm_knowledge_policy.json); dove nessuna soglia raggiunge l'obiettivo,
    quel tipo di input non viene applicato.

Non si passa il nome della gang: non aiuta a identificare la vittima e
spingerebbe il modello a ricordare i tracker ransomware (le stesse fonti).

Modalita':
  validate  eventi con paese E settore osservati, etichette nascoste;
            --per-type eventi per ciascun tipo di input.
  apply     eventi con paese e/o settore ancora mancanti in events_site.json.
"""
import os, re, json, argparse, random
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from openai import OpenAI

MODEL = "openai/gpt-oss-120b"
PROMPT_VERSION = "k1"
BUILD = Path("build")
SECTORS = json.load(open("sector_crosswalk.json", encoding="utf-8"))["target_taxonomy"]

SYSTEM = f"""You identify organisations from your own knowledge. You have NO web access.
Given an organisation name and optional clues, state its HEADQUARTERS COUNTRY and INDUSTRY SECTOR.
- country_iso2: ISO 3166-1 alpha-2 code, or "unknown".
- sector: exactly one of {SECTORS}, or "unknown".
- Answer only if you actually know this organisation, or the clues (domain, description)
  make the answer evident. Do NOT guess from the language or style of the name alone.
  A wrong answer is worse than "unknown".
- country_confidence and sector_confidence: your probability (0-1) that each answer is correct.
Reply with ONLY this JSON object:
{{"country_iso2": "..", "sector": "..", "country_confidence": 0.0, "sector_confidence": 0.0}}"""


def input_type(r):
    if r.get("desc"):
        return "name+desc"
    if r.get("domain"):
        return "name+domain"
    return "name_only"


def user_msg(r):
    parts = [f"Organisation name: {r['victim']}"]
    if r.get("domain"):
        parts.append(f"Website domain: {r['domain']}")
    if r.get("desc"):
        parts.append(f"Description from the data-leak post: {r['desc'][:600]}")
    return "\n".join(parts)


def parse(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    c = str(d.get("country_iso2", "unknown")).strip().upper()
    s = str(d.get("sector", "unknown")).strip()
    return {"country": c if len(c) == 2 and c.isalpha() else None,
            "sector": s if s in SECTORS else None,
            "country_confidence": float(d.get("country_confidence") or 0),
            "sector_confidence": float(d.get("sector_confidence") or 0)}


def load_events():
    src = BUILD / "events_site.json"
    ev = pd.read_json(src)
    keys = pd.read_json(BUILD / "event_keys.json").set_index("event_id")
    ev["desc"] = ev.event_id.map(keys._desc).fillna("").str.strip()
    ev["itype"] = [input_type(r) for r in ev.to_dict("records")]
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["validate", "apply"])
    ap.add_argument("--per-type", type=int, default=200)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--after", default=None,
                    help="validate: solo eventi scoperti da questa data (es. dopo il cutoff di addestramento)")
    a = ap.parse_args()

    ev = load_events()
    if a.mode == "validate":
        pool = ev[(ev.country_src == "observed") & (ev.sector_src == "observed")]
        if a.after:
            pool = pool[pd.to_datetime(pool.discovered) >= a.after]
        rows = pd.concat([g.sample(min(a.per_type, len(g)), random_state=a.seed)
                          for _, g in pool.groupby("itype")])
    else:
        rows = ev[(ev.country_src == "missing") | (ev.sector_src == "missing")]
    out = BUILD / f"llm_knowledge_{a.mode}{'_after' + a.after[:7] if a.after else ''}.jsonl"
    done = {json.loads(l)["event_id"] for l in open(out)} if out.exists() else set()
    todo = [r for r in rows.to_dict("records") if r["event_id"] not in done]
    random.Random(a.seed).shuffle(todo)
    print(f"{a.mode}: {len(rows)} eventi | fatti {len(done)} | da fare {len(todo)} | "
          f"tipi {rows.itype.value_counts().to_dict()}", flush=True)

    client = OpenAI(base_url=os.environ.get("LOCAL_LLM_URL", "http://127.0.0.1:8011/v1"),
                    api_key="local", timeout=180.0, max_retries=2)

    def one(r):
        try:
            resp = client.chat.completions.create(
                model=MODEL, temperature=0, reasoning_effort="low",
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg(r)}])
            raw, tok = resp.choices[0].message.content, resp.usage.total_tokens
            res = {"raw": raw, "tokens": tok, "parsed": parse(raw)}
        except Exception as e:
            res = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        rec = {"event_id": r["event_id"], "itype": r["itype"], "model": MODEL, "backend": "vllm-local",
               "prompt_version": PROMPT_VERSION, **res,
               "need_country": r["country_src"] == "missing", "need_sector": r["sector_src"] == "missing",
               "ts": datetime.now().isoformat(timespec="seconds")}
        if a.mode == "validate":
            rec.update(true_country=r["country"], true_sector=r["sector"],
                       sector_conflict=bool(r["sector_conflict"]), sector_candidates=r["sector_candidates"])
        return rec

    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
        for i, rec in enumerate(ex.map(one, todo), 1):
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n"); f.flush()
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
