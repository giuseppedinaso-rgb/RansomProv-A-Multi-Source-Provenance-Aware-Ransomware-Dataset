#!/usr/bin/env python3
"""Stadio 1c — scarica ransomlook.io per anno/mese.

API pubblica, nessuna chiave (swagger: https://www.ransomlook.io/swagger.json).
Contenuti sotto CC BY 4.0: citare ransomlook.io nel rilascio.

Schema di un post:
  post_title, discovered, group_name, description, link, magnet, misp_uuid
NIENTE paese e NIENTE settore: questa fonte serve a copertura e
corroborazione, non ad arricchire gli attributi.

A fine corsa confronta il totale scaricato con /api/stats.posts_total.
"""
import json, time
from pathlib import Path
from datetime import datetime, date
import requests

RAW = Path("raw"); RAW.mkdir(exist_ok=True)
OUT  = RAW / "ransomlook_raw.json"
CKPT = RAW / ".ransomlook_ckpt.json"

BASE  = "https://www.ransomlook.io/api"
START = (2015, 1)       # prima del primo post noto: i mesi vuoti costano poco


def get_json(url, tries=4, timeout=60, pause=0.4):
    delay = 2.0
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, timeout=timeout, headers={"accept": "application/json"})
        except Exception as e:
            if attempt == tries:
                return None, f"{type(e).__name__}: {e}"
            time.sleep(delay); delay *= 2; continue
        if r.status_code == 200:
            time.sleep(pause)
            try:
                return r.json(), None
            except Exception as e:
                return None, f"JSON non valido: {e}"
        if r.status_code in (429, 500, 502, 503, 504):
            if attempt == tries:
                return None, f"HTTP {r.status_code} dopo {tries} tentativi"
            time.sleep(delay); delay *= 2; continue
        return None, f"HTTP {r.status_code}: {r.text[:120]}"
    return None, "esaurito"


def key(p):
    """Identita' di un post. misp_uuid se c'e', altrimenti la terna."""
    if p.get("misp_uuid"):
        return p["misp_uuid"]
    return f"{p.get('group_name')}|{p.get('post_title')}|{p.get('discovered')}"


def main():
    stats, _ = get_json(f"{BASE}/stats")
    total_api = stats.get("posts_total") if isinstance(stats, dict) else None
    print(f"post dichiarati dall'API: {total_api}", flush=True)

    today = date.today()
    months, (y, m) = [], START
    while (y, m) <= (today.year, today.month):
        months.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    print(f"mesi: {len(months)}  ({months[0]} -> {months[-1]})", flush=True)

    by_key, failures, per_month, done = {}, [], {}, set()
    if CKPT.exists():
        c = json.loads(CKPT.read_text(encoding="utf-8"))
        by_key, failures, per_month, done = c["by_key"], c["failures"], c["per_month"], set(c["done"])
        print(f"ripresa: {len(done)} mesi, {len(by_key)} post", flush=True)

    for i, (yy, mm) in enumerate(months, 1):
        tag = f"{yy}-{mm:02d}"
        if tag in done:
            continue
        data, e = get_json(f"{BASE}/posts/{yy}/{mm:02d}")
        if e:
            failures.append({"year": yy, "month": mm, "error": e})
            print(f"  [{i}/{len(months)}] {tag}  ERRORE {e}", flush=True)
        else:
            posts = data if isinstance(data, list) else (data.get("posts", []) if isinstance(data, dict) else [])
            posts = [p for p in posts if isinstance(p, dict)]
            for p in posts:
                by_key[key(p)] = p
            per_month[tag] = len(posts)
            if posts:
                print(f"  [{i}/{len(months)}] {tag}  +{len(posts)}  totale={len(by_key)}", flush=True)
        done.add(tag)
        CKPT.write_text(json.dumps({"done": sorted(done), "by_key": by_key,
                                    "failures": failures, "per_month": per_month}),
                        encoding="utf-8")

    records = list(by_key.values())
    json.dump({"fetched_at": datetime.now().isoformat(timespec="seconds"),
               "source": "ransomlook.io", "license": "CC BY 4.0",
               "months": [list(months[0]), list(months[-1])],
               "n_records": len(records), "api_total": total_api,
               "per_month": per_month, "failures": failures, "records": records},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False)

    print(f"\nFATTO: {len(records)} post unici", flush=True)
    print(f"  dichiarati dall'API : {total_api}", flush=True)
    if total_api:
        print(f"  copertura           : {len(records)/total_api*100:.1f}%", flush=True)
    print(f"  fallimenti          : {len(failures)}", flush=True)
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
