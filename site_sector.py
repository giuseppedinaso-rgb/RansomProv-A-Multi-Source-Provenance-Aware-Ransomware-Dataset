#!/usr/bin/env python3
"""Stadio 5d — settore dal testo del sito della vittima (LLM senza ricerca web).

site_enrich.py salva per ogni evento un estratto della pagina (titolo, meta
description, intestazioni h1/h2, primi 1500 caratteri del testo), preso dallo
snapshot di archive.org precedente all'attacco quando esiste. Qui un LLM
classifica quell'estratto in una delle 13 classi della tassonomia target.

Perche' senza ricerca: il testo e' gia' la fonte primaria (il sito
dell'organisation stessa), quindi la ricerca non aggiunge evidenza e costa
20-200 volte piu' token. Una chiamata qui usa ~1.000 token.

Modalita':
  validate  eventi di site_validate.jsonl con settore osservato; il settore
            vero NON e' mostrato al modello. Si valuta su due gruppi:
            'agree' (le fonti concordano: etichetta affidabile) e 'conflict'
            (le fonti discordano: si conta corretto se coincide con uno dei
            candidati).
  apply     eventi di site_apply.jsonl con settore mancante e testo scaricato.
"""
import os, re, json, time, argparse, random
from pathlib import Path
from datetime import datetime
import pandas as pd
from groq import Groq

MODEL = "openai/gpt-oss-120b"
PROMPT_VERSION = "s1"
BUILD = Path("build")
SECTORS = json.load(open("sector_crosswalk.json", encoding="utf-8"))["target_taxonomy"]
TPM = int(os.environ.get("GROQ_TPM", "8000"))

SYSTEM = f"""You classify organisations by industry sector from an excerpt of their own website.
Answer with the sector of the organisation's MAIN business activity.
- sector: exactly one of {SECTORS}, or "unknown".
- Use "unknown" if the excerpt does not describe the business (parking page, login page,
  cookie banner only, domain for sale, unrelated content). A wrong answer is worse than "unknown".
- Hints: law firms, accounting, consulting, engineering services, real estate, architecture,
  staffing -> Professional Services; construction and building materials -> Manufacturing;
  software, IT services, telecom -> Technology; banks, insurance -> Financial Services;
  public bodies, municipalities, military -> Government & Defense; schools, universities -> Education;
  hotels, restaurants, travel -> Hospitality; logistics, shipping, automotive dealers' transport -> Transportation;
  non-profits, religious organisations, media, sport -> Other.
- confidence: your probability (0-1) that the sector is correct.
Reply with ONLY this JSON object:
{{"sector": "..", "confidence": 0.0}}"""


def user_msg(name, domain, t):
    return (f"Organisation name: {name}\nDomain: {domain}\n"
            f"Page title: {t.get('title', '')}\nMeta description: {t.get('meta', '')}\n"
            f"Headings: {t.get('heads', '')}\nPage text: {t.get('body', '')}")


def parse(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    s = str(d.get("sector", "unknown")).strip()
    return {"sector": s if s in SECTORS else None, "confidence": float(d.get("confidence") or 0)}


def call(client, msg, tries=20):
    for attempt in range(tries):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=MODEL, temperature=0, reasoning_effort="low",
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}])
            used = resp.usage.prompt_tokens + resp.usage.completion_tokens
            time.sleep(max(0.0, used / TPM * 60 - (time.time() - t0)))
            return {"raw": resp.choices[0].message.content, "tokens": used}
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:300]}"
            low = last.lower()
            if "rate" in low or "429" in low:
                w = re.search(r"try again in (?:(\d+)m)?([\d.]+)(ms|s)", last)
                sec = (int(w.group(1) or 0) * 60 + float(w.group(2)) / (1000 if w.group(3) == "ms" else 1)) if w else 30
                time.sleep(sec + 2)
                continue
            time.sleep(10)
    return {"error": last}


def main():
    global MODEL, TPM
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["validate", "apply"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--shard", default="0/1", help="k/n: solo eventi con event_id %% n == k")
    a = ap.parse_args()
    MODEL = a.model
    k, nsh = map(int, a.shard.split("/"))

    ev = pd.read_json(BUILD / "events.json").set_index("event_id")
    src = BUILD / ("site_validate.jsonl" if a.mode == "validate" else "site_apply.jsonl")
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    rows = [r for r in rows if r.get("fetched") and r.get("text")]
    if a.mode == "validate":
        rows = [r for r in rows if ev.at[r["event_id"], "sector_src"] == "observed"]
        random.Random(a.seed).shuffle(rows)
        rows = rows[:a.n]
    else:
        rows = [r for r in rows if ev.at[r["event_id"], "sector_src"] == "missing"]
    rows = [r for r in rows if r["event_id"] % nsh == k]

    # un file per modello (il 120b mantiene il nome storico); il set dei fatti
    # considera tutti i file, cosi' nessun evento viene classificato due volte
    tag = "" if MODEL == "openai/gpt-oss-120b" else "_" + MODEL.split("/")[-1]
    if os.environ.get("LOCAL_LLM_URL"):
        tag += "_local"
    out = BUILD / f"site_sector_{a.mode}{tag}.jsonl"
    # Groq: il set dei fatti considera tutti i file Groq, cosi' i due shard
    # (120b/20b) non classificano due volte lo stesso evento. Locale: corsa
    # completa e indipendente (validazione e applicazione con un solo modello).
    files = [out] if os.environ.get("LOCAL_LLM_URL") else \
        [f for f in BUILD.glob(f"site_sector_{a.mode}*.jsonl") if "_local" not in f.name]
    done = set()
    for fp in files:
        if fp.exists():
            done |= {json.loads(l)["event_id"] for l in open(fp) if '"error"' not in l}
    todo = [r for r in rows if r["event_id"] not in done]
    print(f"{a.mode}: {len(rows)} eventi | fatti {len(done)} | da fare {len(todo)}", flush=True)

    # LOCAL_LLM_URL: server OpenAI-compatibile locale (vllm) con gli stessi pesi
    # pubblici del modello; senza limiti di token. Altrimenti Groq.
    local = os.environ.get("LOCAL_LLM_URL")
    if local:
        from openai import OpenAI
        client = OpenAI(base_url=local, api_key="local", timeout=120.0, max_retries=2)
        TPM = 10**9                                   # nessun pacing in locale
    else:
        client = Groq(api_key=os.environ["GROQ_API_KEY"], timeout=60.0, max_retries=0)
    backend = "vllm-local" if local else "groq"
    with open(out, "a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            e = r["event_id"]
            res = call(client, user_msg(ev.at[e, "victim"], r["domain"], r["text"]))
            rec = {"event_id": e, "model": MODEL, "backend": backend, "prompt_version": PROMPT_VERSION,
                   "origin": r.get("origin"), **res, "parsed": parse(res.get("raw"))}
            if a.mode == "validate":
                rec["true_sector"] = ev.at[e, "sector"]
                rec["sector_conflict"] = bool(ev.at[e, "sector_conflict"])
                rec["sector_candidates"] = ev.at[e, "sector_candidates"]
            rec["ts"] = datetime.now().isoformat(timespec="seconds")
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n"); f.flush()
            p = rec["parsed"] or {}
            print(f"  {i}/{len(todo)} tok={res.get('tokens')} -> {p.get('sector')} ({p.get('confidence')})"
                  + (f" | vero {rec.get('true_sector')}" if a.mode == "validate" else ""), flush=True)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
