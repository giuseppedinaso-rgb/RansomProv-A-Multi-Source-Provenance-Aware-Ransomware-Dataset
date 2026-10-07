#!/usr/bin/env python3
"""Stadio 5 — paese e settore mancanti con un LLM dotato di ricerca web.

Modello: openai/gpt-oss-120b su Groq, strumento integrato `browser_search`.

Due modalita':
  validate  campione di eventi con paese E settore noti e unanimi; le
            etichette vengono nascoste e confrontate con la risposta.
  apply     eventi con paese e/o settore mancante.

Scelte di progetto:
  * output vincolato: paese ISO alpha-2, settore da lista chiusa (le 13
    classi del target), confidenza, possibilita' esplicita di 'unknown';
  * i TRACKER RANSOMWARE sono vietati come fonte: sono le stesse fonti che
    aggreghiamo. In validazione il modello ritroverebbe le etichette
    nascoste (accuratezza gonfiata); in applicazione riprodurrebbe gli stessi
    errori delle fonti. Una risposta la cui evidenza e' un tracker viene
    scartata come 'tracker_evidence';
  * ogni risposta e' salvata grezza in un .jsonl con modello, versione del
    prompt, data, numero di ricerche e token: la corsa si riprende da dove si
    e' fermata e ogni etichetta resta verificabile.
"""
import os, re, json, time, argparse, random
from pathlib import Path
from datetime import datetime
import pandas as pd
from groq import Groq

MODEL = "openai/gpt-oss-120b"
PROMPT_VERSION = "v3"
BUILD = Path("build")
SECTORS = json.load(open("sector_crosswalk.json", encoding="utf-8"))["target_taxonomy"]

TRACKERS = ["ransomware.live", "ransomfeed.it", "ransomlook.io", "ransomwatch", "ecrime.ch",
            "hackmanac", "darkfeed", "falconfeeds", "ransomwarefeed", "cybernews.com/ransomware",
            "breachsense", "redpacketsecurity", "watchguard.com/wgrd-security-hub/ransomware",
            "comparitech.com/ransomware", "dexpose", "hookphish", "halcyon.ai/attacks",
            "ransomware-tracker", "deepwebwatch", "socradar"]

SYSTEM = f"""You identify organisations. Given a company name and optional clues, use web search
to find the organisation's HEADQUARTERS COUNTRY and its INDUSTRY SECTOR.

Rules:
- Use AT MOST 3 web searches. Start with the website domain if one is given,
  otherwise search the exact company name. Stop as soon as you have identified it.
- Base the answer on the organisation's own website, business registries, LinkedIn,
  or reputable news. Do NOT use ransomware or data-leak tracking sites as evidence
  ({", ".join(TRACKERS[:8])}, and similar).
- country_iso2: ISO 3166-1 alpha-2 code, or "unknown".
- sector: exactly one of {SECTORS}, or "unknown".
- If you cannot identify the specific organisation with reasonable certainty, answer "unknown".
  A wrong answer is worse than "unknown".
- confidence: your probability (0-1) that BOTH country and sector are correct.
The ONLY tool you may call is browser_search. Never call any other tool: there is no
"json" tool. When you are done searching, write your final answer as plain text
containing ONLY this JSON object and nothing else:
{{"country_iso2": "..", "sector": "..", "confidence": 0.0, "evidence_url": "..", "matched_name": ".."}}"""


def user_msg(r):
    parts = [f"Company name: {r['victim']}"]
    if r.get("domain"):
        parts.append(f"Website/domain: {r['domain']}")
    if r.get("desc"):
        parts.append(f"Description from a leak post: {r['desc'][:600]}")
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
    return {"country": None if c in ("UNKNOWN", "", "NONE") or len(c) != 2 else c,
            "sector": s if s in SECTORS else None,
            "confidence": float(d.get("confidence") or 0),
            "evidence_url": d.get("evidence_url"), "matched_name": d.get("matched_name")}


def is_tracker(url):
    u = str(url or "").lower()
    return any(t in u for t in TRACKERS)


TPM = int(os.environ.get("GROQ_TPM", "8000"))     # limite token/minuto del piano


def call(client, r, search=True, tries=30):
    """Una chiamata, con ritmo imposto dal limite di token al minuto.

    Una chiamata con ricerca consuma ~10.000 token contro un limite di 8.000
    al minuto sul piano gratuito: ritentare a raffica non serve. Dopo ogni
    risposta si attende il tempo che quei token richiedono (pacing), e sui
    429 si attende almeno fino al ricarico del secchio.
    """
    last, temp = "", 0.0
    for attempt in range(tries):
        t0 = time.time()
        try:
            kw = dict(model=MODEL, temperature=temp,
                      messages=[{"role": "system", "content": SYSTEM},
                                {"role": "user", "content": user_msg(r)}])
            if search:
                kw["tools"] = [{"type": "browser_search"}]
            resp = client.chat.completions.create(**kw)
            m = resp.choices[0].message
            et = getattr(m, "executed_tools", None) or []
            used = resp.usage.prompt_tokens + resp.usage.completion_tokens
            time.sleep(max(0.0, used / TPM * 60 - (time.time() - t0)))
            return {"raw": m.content, "n_searches": len(et),
                    "tokens_in": resp.usage.prompt_tokens, "tokens_out": resp.usage.completion_tokens}
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:300]}"
            low = last.lower()
            if "tool call validation failed" in low:
                # il modello ha inventato uno strumento: a temperatura 0 si
                # ripeterebbe identico, quindi si ritenta con un po' di varieta'
                temp = min(0.2 + 0.2 * attempt, 0.8)
                time.sleep(15)
                if attempt >= 5:
                    return {"error": last}
                continue
            if "timeout" in low or "timed out" in low:
                print(f"    timeout, ritento ({attempt+1})", flush=True)
                time.sleep(20)
                if attempt >= 3:
                    return {"error": last}
                continue
            if "rate" in low or "429" in low or "413" in low or "too large" in low:
                w = re.search(r"try again in (?:(\d+)m)?([\d.]+)(ms|s)", last)
                sec = 0.0
                if w:
                    sec = (int(w.group(1) or 0) * 60 + float(w.group(2)) / (1000 if w.group(3) == "ms" else 1))
                time.sleep(max(sec + 2, 30))           # almeno mezzo minuto di ricarico
                continue
            if attempt >= 3:
                return {"error": last}
            time.sleep(10 * (attempt + 1))
    return {"error": f"esaurito dopo {tries} tentativi: {last}"}


def load_events():
    ev = pd.read_json(BUILD / "events.json")
    u = pd.read_json(BUILD / "unified.json")
    m = pd.read_json(BUILD / "record_to_event.json"); u["event_id"] = m.event_id.values
    desc = u.groupby("event_id").description.agg(
        lambda x: max([str(v) for v in x if v], key=len, default=""))
    ev["desc"] = ev.event_id.map(desc).fillna("")
    ev["input_type"] = ["name+desc" if len(d) >= 50 else ("name+domain" if dom else "name_only")
                        for d, dom in zip(ev.desc, ev.domain)]
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["validate", "apply"])
    ap.add_argument("--n", type=int, default=450, help="validate: dimensione del campione")
    ap.add_argument("--no-search", action="store_true", help="confronto: stesso modello senza ricerca")
    ap.add_argument("--limit", type=int, default=None, help="massimo di chiamate in questa corsa")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", default="llm_policy.json",
                    help="apply: file prodotto da llm_policy.py che dice dove usare la ricerca")
    a = ap.parse_args()

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit("esporta GROQ_API_KEY")
    # timeout esplicito: una ricerca che non torna non deve bloccare la corsa
    client = Groq(api_key=key, timeout=180.0, max_retries=0)
    ev = load_events()
    tag = "nosearch" if a.no_search else "search"

    if a.mode == "validate":
        pool = ev[(ev.country_resolution == "unanimous") & (ev.sector_resolution == "unanimous")]
        # stratificato per tipo di input, in proporzioni uguali: i mancanti sono
        # per meta' 'solo nome', e li' la ricerca conta di piu'
        per = a.n // 3
        sample = pd.concat([g.sample(min(per, len(g)), random_state=a.seed)
                            for _, g in pool.groupby("input_type")])
        out = BUILD / f"llm_validate_{tag}.jsonl"
    else:
        sample = ev[(ev.country_src == "missing") | (ev.sector_src == "missing")]
        out = BUILD / "llm_apply.jsonl"
        pol = json.load(open(a.policy, encoding="utf-8"))
        search_for = set(pol["search_for_input_types"])
        print(f"policy: ricerca per {sorted(search_for)}, senza ricerca per il resto", flush=True)

    done = set()
    if out.exists():
        # contano solo i record riusciti con la versione di prompt corrente:
        # una validazione deve misurare un prompt solo
        done = {r["event_id"] for r in map(json.loads, open(out, encoding="utf-8"))
                if "error" not in r and r.get("prompt_version") == PROMPT_VERSION}
    todo = [r for r in sample.to_dict("records") if r["event_id"] not in done]
    # ordine mescolato e riproducibile: ogni risultato parziale copre tutte le
    # fasce di input, invece di esaurirne una prima di passare alla successiva
    random.Random(a.seed).shuffle(todo)
    if a.limit:
        todo = todo[: a.limit]
    print(f"{a.mode}/{tag}: campione {len(sample)} | gia' fatti {len(done)} | in questa corsa {len(todo)}", flush=True)

    with open(out, "a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            use_search = (r["input_type"] in search_for) if a.mode == "apply" else not a.no_search
            res = call(client, r, search=use_search)
            p = parse(res.get("raw")) if "raw" in res else None
            rec = {"event_id": r["event_id"], "input_type": r["input_type"],
                   "victim": r["victim"], "domain": r.get("domain"),
                   "true_country": r["country"] if a.mode == "validate" else None,
                   "true_sector": r["sector"] if a.mode == "validate" else None,
                   "model": MODEL, "search": use_search, "prompt_version": PROMPT_VERSION,
                   "need_country": r.get("country_src") == "missing" if a.mode == "apply" else True,
                   "need_sector": r.get("sector_src") == "missing" if a.mode == "apply" else True,
                   "ts": datetime.now().isoformat(timespec="seconds"), **res, "parsed": p}
            if p and is_tracker(p.get("evidence_url")):
                rec["rejected"] = "tracker_evidence"
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            pp = rec.get("parsed") or {}
            print(f"  {i}/{len(todo)} {rec['ts'][11:]} {r['input_type']:11} "
                  f"ric={rec.get('n_searches')} tok={(rec.get('tokens_in') or 0)+(rec.get('tokens_out') or 0)} "
                  f"-> {pp.get('country')}/{pp.get('sector')} conf={pp.get('confidence')} "
                  f"{('ERR '+rec['error'][:80]) if 'error' in rec else ''}", flush=True)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
