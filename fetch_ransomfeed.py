#!/usr/bin/env python3
"""Stadio 1 — ransomfeed.it, con partizionamento adattivo.

LIMITE DELL'API, misurato:
  * /gang/{g}/date/{y}/offset/{n} restituisce al massimo n record;
  * n > 1000 NON pagina: il server ripiega al default di 100. Il massimo
    utile e' quindi esattamente 1000;
  * non esiste cursore, `page`, `start` o filtro per mese;
  * `country` e' un filtro valido e si compone: /gang/{g}/date/{y}/country/{c}.

Strategia: se una coppia gang-anno restituisce esattamente 1000 record e'
troncata, quindi la si rispezza per paese. Se anche una terzina gang-anno-paese
satura, viene registrata in `still_saturated`: un buco noto, non silenzioso.
"""
import json, time, urllib.parse
from pathlib import Path
from datetime import datetime, date
import requests

RAW = Path("raw"); RAW.mkdir(exist_ok=True)
OUT  = RAW / "ransomfeed_raw.json"
CKPT = RAW / ".ransomfeed_ckpt.json"

BASE  = "https://api.ransomfeed.it"
START = 2018
END   = date.today().year
LIMIT = 1000                     # massimo utile: oltre, l'API torna a 100


def get_json(url, tries=4, timeout=30, pause=0.2):
    delay = 1.0
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, timeout=timeout)
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


Q = lambda s: urllib.parse.quote(str(s), safe="")


def main():
    stats, _ = get_json(f"{BASE}/stats")
    gangs, e1 = get_json(f"{BASE}/list/gang")
    countries, e2 = get_json(f"{BASE}/list/country")
    if e1 or e2:
        raise SystemExit(f"liste non recuperate: {e1 or e2}")
    total_api = stats.get("total") if isinstance(stats, dict) else None
    print(f"gang {len(gangs)} | paesi {len(countries)} | anni {START}-{END} | "
          f"totale dichiarato dall'API: {total_api}", flush=True)

    by_id, failures, split, still_sat, done = {}, [], [], [], set()
    if CKPT.exists():
        c = json.loads(CKPT.read_text(encoding="utf-8"))
        by_id = {int(k): v for k, v in c["by_id"].items()}
        failures, split, still_sat = c["failures"], c["split"], c["still_saturated"]
        done = set(c["done"])
        print(f"ripresa: {len(done)} coppie fatte, {len(by_id)} record unici", flush=True)

    def absorb(recs):
        for r in recs:
            rid = r.get("id")
            if rid is not None:
                by_id[int(rid)] = r

    todo = [(g, y) for g in gangs for y in range(START, END + 1)]
    t0 = time.time()
    for n, (gang, year) in enumerate(todo, 1):
        tag = f"{gang}|{year}"
        if tag in done:
            continue
        data, e = get_json(f"{BASE}/gang/{Q(gang)}/date/{year}/offset/{LIMIT}")
        if e:
            failures.append({"gang": gang, "year": year, "error": e})
        elif isinstance(data, list) and data:
            if len(data) < LIMIT:
                absorb(data)
            else:
                # troncata: rispezza per paese
                absorb(data)                      # tienili comunque
                split.append({"gang": gang, "year": year})
                got = 0
                for c in countries:
                    d2, e2 = get_json(
                        f"{BASE}/gang/{Q(gang)}/date/{year}/country/{Q(c)}/offset/{LIMIT}")
                    if e2:
                        failures.append({"gang": gang, "year": year, "country": c, "error": e2})
                        continue
                    if isinstance(d2, list) and d2:
                        absorb(d2); got += len(d2)
                        if len(d2) >= LIMIT:
                            still_sat.append({"gang": gang, "year": year, "country": c})
                print(f"    split {gang}/{year}: {got} record da {len(countries)} paesi", flush=True)
        done.add(tag)
        if n % 100 == 0:
            print(f"[{n}/{len(todo)}] unici={len(by_id)} errori={len(failures)} "
                  f"split={len(split)} {(time.time()-t0)/60:.1f}min", flush=True)
            CKPT.write_text(json.dumps({"done": sorted(done), "by_id": by_id,
                                        "failures": failures, "split": split,
                                        "still_saturated": still_sat}), encoding="utf-8")

    records = list(by_id.values())
    json.dump({"fetched_at": datetime.now().isoformat(timespec="seconds"),
               "source": "ransomfeed.it", "years": [START, END],
               "n_gangs": len(gangs), "n_records": len(records),
               "api_total": total_api, "limit_per_request": LIMIT,
               "failures": failures, "split_by_country": split,
               "still_saturated": still_sat, "records": records},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False)

    print(f"\nFATTO: {len(records)} record unici", flush=True)
    print(f"  dichiarati dall'API : {total_api}", flush=True)
    if total_api:
        print(f"  copertura           : {len(records)/total_api*100:.1f}%", flush=True)
    print(f"  coppie rispezzate   : {len(split)}", flush=True)
    print(f"  ancora sature       : {len(still_sat)}", flush=True)
    print(f"  fallimenti          : {len(failures)}", flush=True)
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
