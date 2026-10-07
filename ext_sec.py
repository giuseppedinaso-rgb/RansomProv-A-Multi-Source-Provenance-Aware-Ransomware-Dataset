#!/usr/bin/env python3
"""Validazione esterna con le notifiche SEC 8-K Item 1.05 -> build/ext_sec.json.

Dal 18 dicembre 2023 le societa' quotate negli USA devono notificare un
incidente informatico rilevante con un modulo 8-K, Item 1.05. Si scaricano dalla
ricerca full-text di EDGAR (efts.sec.gov) i moduli 8-K / 8-K/A che contengono
'ransomware' e dichiarano l'Item 1.05: sono attacchi ransomware CONFERMATI
dalla vittima stessa, indipendenti dai leak site.

Misure:
  * copertura: quota di questi incidenti presenti nel dataset (una vittima
    confermata compare tra le rivendicazioni?). Non tutte le vittime finiscono
    su un leak site (chi paga, gang senza leak site): e' una stima per difetto
    della copertura ma un buon test di ricall su un insieme verificato;
  * ritardo: giorni tra la rivendicazione sul leak site e la notifica SEC.

Abbinamento: nome societa' normalizzato (senza forme societarie e
punteggiatura) uguale alla chiave della vittima, oppure contenuto in essa, o
similarita' >= 0.92; con la rivendicazione entro -180/+180 giorni dalla
notifica. Gli abbinamenti sono salvati in audit/sec_matches.csv per una
verifica manuale.
"""
import json, re, time
from pathlib import Path
from difflib import SequenceMatcher
import requests
import pandas as pd

BUILD, AUDIT = Path("build"), Path("audit")
URL = "https://efts.sec.gov/LATEST/search-index"
UA = {"User-Agent": "Academic research ransomware-dataset admin@example.org"}
CORP = r"\b(inc|incorporated|corp|corporation|co|company|ltd|limited|llc|plc|holdings?|group|the|sa|nv|ag|se|lp)\b"


def norm(s):
    s = re.sub(r"\(.*?\)", " ", str(s).lower())
    s = re.sub(r"\.(com|net|org|io|co)\b", " ", s)
    s = re.sub(CORP, " ", s)
    return re.sub(r"[^a-z0-9]", "", s)


def fetch():
    hits, frm = [], 0
    while True:
        r = requests.get(URL, params={"q": '"Item 1.05"', "forms": "8-K",
                                      "startdt": "2023-12-18", "enddt": "2026-09-30", "from": frm},
                         headers=UA, timeout=60)
        r.raise_for_status()
        j = r.json()["hits"]
        h = j["hits"]
        hits += h; frm += len(h)
        if not h or frm >= j["total"]["value"]:          # oltre il totale EDGAR risponde 500
            break
        time.sleep(0.3)
    rows = []
    for x in hits:
        s = x["_source"]
        if "1.05" not in (s.get("items") or []):
            continue
        name = re.sub(r"\s*\(.*$", "", s["display_names"][0]).strip()
        rows.append({"company": name, "cik": s.get("ciks", [None])[0], "file_date": s["file_date"],
                     "form": s.get("form"), "adsh": s.get("adsh")})
    df = pd.DataFrame(rows).sort_values("file_date")
    # un incidente per societa': prima notifica (gli 8-K/A successivi sono aggiornamenti)
    return df.drop_duplicates("cik", keep="first").reset_index(drop=True), len(hits)


def main():
    AUDIT.mkdir(exist_ok=True)
    sec, n_hits = fetch()
    ev = pd.read_json("release/events_enriched.json")
    ev["d"] = pd.to_datetime(ev.discovered)
    ev["k1"] = ev.victim.map(norm)
    ev["k2"] = ev.domain.fillna("").map(lambda d: norm(d.split(".")[0]) if d else "")
    matches = []
    for _, s in sec.iterrows():
        k, fd = norm(s.company), pd.Timestamp(s.file_date)
        if len(k) < 3:
            continue
        w = ev[(ev.d >= fd - pd.Timedelta(days=180)) & (ev.d <= fd + pd.Timedelta(days=180))]
        best = None
        for _, e in w.iterrows():
            for kk in (e.k1, e.k2):
                if not kk or len(kk) < 3:
                    continue
                # abbinamento severo: chiave identica, oppure una contenuta
                # nell'altra con almeno 6 caratteri e rapporto di lunghezza >= 0.6,
                # oppure similarita' >= 0.92
                short, long_ = sorted((k, kk), key=len)
                if kk == k:
                    sim = 1.0
                elif len(short) >= 6 and short in long_ and len(short) / len(long_) >= 0.6:
                    sim = 0.95
                else:
                    sim = SequenceMatcher(None, k, kk).ratio()
                if sim >= 0.92 and (best is None or sim > best[0]):
                    best = (sim, e)
        if best:
            e = best[1]
            matches.append({"company": s.company, "file_date": s.file_date, "event_id": e.event_id,
                            "victim": e.victim, "domain": e.domain, "gang": e.gang, "discovered": e.discovered,
                            "lag_days": (fd - e.d).days, "similarity": round(best[0], 2), "label": ""})
    m = pd.DataFrame(matches)
    m.to_csv(AUDIT / "sec_matches.csv", index=False)
    sec.to_csv(BUILD / "ext_sec_filings.csv", index=False)
    rep = {"edgar_hits_ransomware_8k": n_hits, "item_105_companies": len(sec),
           "matched_in_dataset": len(m), "coverage_pct": round(len(m) / len(sec) * 100, 1) if len(sec) else None}
    if len(m):
        rep["lag_days_sec_minus_claim"] = {"median": float(m.lag_days.median()),
                                           "iqr": [float(m.lag_days.quantile(.25)), float(m.lag_days.quantile(.75))],
                                           "claim_before_filing_pct": round((m.lag_days > 0).mean() * 100, 1)}
        rep["matched_country_US_pct"] = round((ev.set_index("event_id").loc[m.event_id].country == "US").mean() * 100, 1)
    json.dump(rep, open(BUILD / "ext_sec.json", "w"), indent=2)
    print(json.dumps(rep, indent=2))
    print(m[["company", "victim", "gang", "file_date", "discovered", "lag_days", "similarity"]].to_string())


if __name__ == "__main__":
    main()
