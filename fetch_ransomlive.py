#!/usr/bin/env python3
"""Stadio 2 — scarica ransomware.live per anno/mese, senza filtro di settore."""
import os, json, time, urllib.parse
from pathlib import Path
from datetime import datetime, date
import requests

RAW = Path("raw"); RAW.mkdir(exist_ok=True)
OUT  = RAW / "ransomlive_raw.json"
CKPT = RAW / ".ransomlive_ckpt.json"

BASE  = "https://api-pro.ransomware.live"
KEY   = os.environ.get("RANSOMLIVE_API_KEY", "")
HEAD  = {"accept": "application/json", "X-API-KEY": KEY}
START = (2018, 1)


def get_json(url, headers=None, tries=4, timeout=30, pause=0.25):
    delay = 1.0
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
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


def main():
    if not KEY:
        raise SystemExit("esporta RANSOMLIVE_API_KEY prima di eseguire")
    today = date.today()
    months, (y, m) = [], START
    while (y, m) <= (today.year, today.month):
        months.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    print(f"mesi da scaricare: {len(months)}  ({months[0]} -> {months[-1]})", flush=True)

    records, failures, done = [], [], set()
    if CKPT.exists():
        c = json.loads(CKPT.read_text(encoding="utf-8"))
        records, failures, done = c["records"], c["failures"], set(c["done"])
        print(f"ripresa dal checkpoint: {len(done)} mesi, {len(records)} record", flush=True)

    for i, (yy, mm) in enumerate(months, 1):
        tag = f"{yy}-{mm:02d}"
        if tag in done:
            continue
        q = urllib.parse.urlencode({"year": str(yy), "month": f"{mm:02d}"}, safe="%20")
        data, e = get_json(f"{BASE}/victims/?{q}", headers=HEAD)
        if e:
            failures.append({"year": yy, "month": mm, "error": e})
            print(f"  [{i}/{len(months)}] {tag}  ERRORE {e}", flush=True)
        else:
            v = data.get("victims", []) if isinstance(data, dict) else data
            v = [x for x in v if isinstance(x, dict)]
            records.extend(v)
            print(f"  [{i}/{len(months)}] {tag}  +{len(v)}  totale={len(records)}", flush=True)
        done.add(tag)
        CKPT.write_text(json.dumps({"done": sorted(done), "records": records,
                                    "failures": failures}), encoding="utf-8")

    json.dump({"fetched_at": datetime.now().isoformat(timespec="seconds"),
               "source": "ransomware.live",
               "months": [list(months[0]), list(months[-1])],
               "n_records": len(records), "failures": failures, "records": records},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"\nFATTO: {len(records)} record | fallimenti {len(failures)}", flush=True)
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
