#!/usr/bin/env python3
"""LLM come modello di previsione del prossimo evento (protocollo nextevent_data).

Modello: openai/gpt-oss-20b, pesi pubblici, servito in locale con vLLM su una GPU
(temperature 0, reasoning_effort low). Nessun addestramento: zero-shot oppure
few-shot (3 esempi fissi presi dal periodo di training).

Per ogni evento del TEST (luglio 2025 - settembre 2026, cioe' dopo il limite
di addestramento del modello, giugno 2024: niente contaminazione dei bersagli)
il prompt contiene il nome della gang e le sue ultime CONTEXT rivendicazioni
(data, paese, settore; 'unknown' dove manca) con le etichette della versione
scelta. Il modello restituisce le probabilita' per:
  interval  same_day / le_7d / gt_7d
  country   fino a 5 codici ISO con probabilita' (le classi non elencate si
            dividono la probabilita' residua)
  sector    fino a 3 classi con probabilita'
Le predizioni sono mappate sul vocabolario del protocollo (paesi fuori dai 30
piu' frequenti -> 'other') e valutate con le stesse metriche degli altri
modelli, comprese log-loss e top-3.

Uso: python nextevent_llm.py <observed|enriched> <zeroshot|fewshot> [max_eventi]
"""
import os, sys, json, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from openai import OpenAI
from nextevent_data import load, targets, metrics, FIELDS, IGNORE, ROOT

OUT = ROOT / "benchmark" / "results" / "nextevent"
RAW = ROOT / "benchmark" / "results" / "nextevent_llm_raw"
MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-20b")
TAG = MODEL.split("/")[-1]
CONTEXT = 30
URL = os.environ.get("LOCAL_LLM_URL", "http://127.0.0.1:8012/v1")


def system_prompt(vocab):
    return f"""You forecast the next ransomware data-leak-site claim of a criminal group.
You get the group's name and its most recent claims (date, victim country, victim sector),
oldest first. Predict the NEXT claim of the same group.
- interval: days from the last listed claim to the next one: "same_day" (0 days),
  "le_7d" (1-7 days) or "gt_7d" (more than 7 days).
- country: ISO 3166-1 alpha-2 code of the next victim.
- sector: one of {vocab['sector'][:-1]}.
Give calibrated probabilities. Reply with ONLY this JSON:
{{"interval": {{"same_day": p, "le_7d": p, "gt_7d": p}},
  "country": {{"<ISO2>": p, ... up to 5 codes}},
  "sector": {{"<sector>": p, ... up to 3 sectors}}}}"""


def history_text(ev, rows, gang):
    lines = [f"Group: {gang}", "Recent claims (date | country | sector):"]
    for r in rows:
        c = ev.country.values[r] if isinstance(ev.country.values[r], str) else "unknown"
        s = ev.sector.values[r] if isinstance(ev.sector.values[r], str) else "unknown"
        lines.append(f"{ev.d.values[r].astype('datetime64[D]')} | {c} | {s}")
    return "\n".join(lines)


def fewshot_messages(ev, idx_by_gang):
    """3 esempi fissi dal training (gang diverse, risposte = valori veri)."""
    msgs, rng = [], np.random.default_rng(0)
    cands = [g for g, ix in idx_by_gang.items() if (ev.split.values[ix] == "train").sum() > 40]
    for g in rng.choice(sorted(cands), 3, replace=False):
        ix = [i for i in idx_by_gang[g] if ev.split.values[i] == "train"]
        k = len(ix) // 2
        hist, nxt = ix[max(0, k - CONTEXT):k], ix[k]
        dd = (ev.d.values[nxt] - ev.d.values[hist[-1]]).astype("timedelta64[D]").astype(int)
        iv = "same_day" if dd == 0 else ("le_7d" if dd <= 7 else "gt_7d")
        ans = {"interval": {iv: 1.0}, "country": {str(ev.country.values[nxt]): 1.0}, "sector": {str(ev.sector.values[nxt]): 1.0}}
        msgs += [{"role": "user", "content": history_text(ev, hist, g)}, {"role": "assistant", "content": json.dumps(ans)}]
    return msgs


def parse(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def to_proba(d, vocab):
    """Distribuzioni sul vocabolario del protocollo; massa residua uniforme."""
    out = {}
    spec = {"delta": ("interval", {"same_day": 1, "le_7d": 2, "gt_7d": 3}),
            "country": ("country", None), "sector": ("sector", None)}
    for f, (key, fixed) in spec.items():
        K = len(vocab[f]); P = np.zeros(K)
        valid = [1, 2, 3] if f == "delta" else list(range(K - 1))           # niente 'first' / 'unknown'
        for lab, p in ((d or {}).get(key) or {}).items():
            try:
                p = float(p)
            except (TypeError, ValueError):
                continue
            if fixed is not None:
                j = fixed.get(lab)
            elif f == "country":
                code = str(lab).upper()[:2]
                j = vocab[f].index(code) if code in vocab[f][:-2] else vocab[f].index("other")
            else:
                j = vocab[f].index(lab) if lab in vocab[f][:-1] else None
            if j is not None and p > 0:
                P[j] += p
        s = P[valid].sum()
        if s > 1: P[valid] /= s; s = 1.0
        rest = [j for j in valid if P[j] == 0]
        P[rest] += (1 - s) / max(len(rest), 1) if rest else 0
        if P.sum() == 0: P[valid] = 1 / len(valid)
        out[f] = P / P.sum()
    return out


def main():
    version, mode = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    ev, vocab = load(version); tgt = targets(ev, vocab)
    idx_by_gang = {g: list(x.index) for g, x in ev.groupby("gang", sort=False)}
    pos = {i: k for ix in idx_by_gang.values() for k, i in enumerate(ix)}
    test = [i for i in ev.index if ev.split.values[i] == "test" and pos[i] > 0
            and any(tgt[f][i] != IGNORE for f in FIELDS)]
    if limit: test = test[:limit]
    sysmsg = [{"role": "system", "content": system_prompt(vocab)}]
    shots = fewshot_messages(ev, idx_by_gang) if mode == "fewshot" else []
    client = OpenAI(base_url=URL, api_key="local", timeout=300, max_retries=3)
    RAW.mkdir(parents=True, exist_ok=True)
    raw_path = RAW / (f"{version}_{mode}.jsonl" if TAG == "gpt-oss-20b" else f"{TAG}_{version}_{mode}.jsonl")
    done = {}
    if raw_path.exists():
        for l in open(raw_path):
            r = json.loads(l); done[r["row"]] = r

    def ask(i):
        g = ev.gang.values[i]; ix = idx_by_gang[g]; k = pos[i]
        hist = ix[max(0, k - CONTEXT):k]
        msgs = sysmsg + shots + [{"role": "user", "content": history_text(ev, hist, g)}]
        try:
            r = client.chat.completions.create(model=MODEL, messages=msgs, temperature=0, reasoning_effort="low", max_tokens=1500)
            return {"row": int(i), "event_id": int(ev.event_id.values[i]), "raw": r.choices[0].message.content,
                    "tokens": r.usage.total_tokens}
        except Exception as e:
            return {"row": int(i), "event_id": int(ev.event_id.values[i]), "error": str(e)[:200]}

    todo = [i for i in test if i not in done]
    print(f"{version} {mode}: {len(test)} eventi, da fare {len(todo)}", flush=True)
    with open(raw_path, "a") as fh, ThreadPoolExecutor(32) as ex:
        for n, r in enumerate(ex.map(ask, todo), 1):
            fh.write(json.dumps(r) + "\n"); done[r["row"]] = r
            if n % 500 == 0: print(f"  {n}/{len(todo)}", flush=True)
    probs = {f: [] for f in FIELDS}; ok = 0
    for i in test:
        d = parse(done[i].get("raw")); ok += d is not None
        P = to_proba(d, vocab)
        for f in FIELDS: probs[f].append(P[f])
    res = {"model": f"{TAG} ({mode})", "version": version, "seed": 0, "n_events": len(test),
           "parsed_pct": ok / len(test), "test": {}}
    for f in FIELDS:
        P = np.vstack(probs[f]); y = tgt[f][test]
        res["test"][f] = metrics(y, P.argmax(1), P)
    json.dump(res, open(OUT / f"{TAG}-{mode}_{version}_s0.json", "w"), indent=2)
    print(json.dumps({f: {k: round(v, 3) for k, v in res["test"][f].items() if k != "n"} for f in FIELDS}))


if __name__ == "__main__":
    main()
