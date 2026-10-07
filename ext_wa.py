#!/usr/bin/env python3
"""Validazione esterna con le notifiche di violazione dello Stato di Washington.

Fonte: Washington State Office of the Attorney General, "Data Breach
Notifications Affecting Washington Residents" (data.wa.gov, dataset sb4j-ca4h,
dati aperti). Ogni notifica riporta nome dell'organizzazione, data di inizio
della violazione (DateStart), data in cui l'organizzazione se ne e' accorta
(DateAware), data di invio (DateSubmitted), settore e tipo di attacco
(CyberattackType). Le notifiche con CyberattackType = Ransomware sono attacchi
ransomware CONFERMATI dalla vittima, indipendenti dai leak site.

Misure (build/ext_wa.json, audit/wa_matches.csv):
  * copertura: quota delle notifiche ransomware abbinate a un evento del dataset
    (rivendicazione tra 30 giorni prima e 365 giorni dopo DateStart), per anno.
    Non tutte le vittime finiscono su un leak site (chi paga), quindi e' un
    limite inferiore della copertura dei leak site;
  * ritardi: rivendicazione - inizio attacco; notifica - rivendicazione;
  * controllo negativo: stessa procedura sulle notifiche NON ransomware
    (phishing, malware...): un abbinamento li' indica un falso positivo del
    metodo o un attacco ransomware classificato diversamente.
"""
import json
from pathlib import Path
import requests
import pandas as pd
from ext_match import prepare, match

BUILD, AUDIT = Path("build"), Path("audit")
URL = "https://data.wa.gov/resource/sb4j-ca4h.json"


def main():
    AUDIT.mkdir(exist_ok=True)
    rows = requests.get(URL, params={"$limit": 50000}, timeout=120).json()
    wa = pd.DataFrame(rows)
    wa["start"] = pd.to_datetime(wa.datestart, errors="coerce")
    wa["submitted"] = pd.to_datetime(wa.datesubmitted, errors="coerce")
    wa["ref"] = wa.start.fillna(wa.submitted)
    wa = wa[(wa.ref >= "2020-06-01") & (wa.ref <= "2026-06-30")]
    wa.to_csv(BUILD / "ext_wa_raw.csv", index=False)
    ev = prepare(pd.read_json("release/events_enriched.json"))

    out, rep = [], {"wa_notifications_2020_2026": len(wa),
                    "by_attack_type": wa.cyberattacktype.fillna("none").value_counts().to_dict()}
    for kind, sub in (("ransomware", wa[wa.cyberattacktype == "Ransomware"]),
                      ("other_cyberattack", wa[wa.cyberattacktype.notna() & (wa.cyberattacktype != "Ransomware")])):
        sub = sub.drop_duplicates("name")
        hits = []
        for r in sub.itertuples():
            m = match(r.name, r.ref, ev, before=30, after=365)
            if m:
                e = m[1]
                hits.append({"kind": kind, "name": r.name, "industry": r.industrytype, "date_start": r.start.date(),
                             "date_submitted": r.submitted.date() if pd.notna(r.submitted) else None,
                             "event_id": e.event_id, "victim": e.victim, "domain": e.domain, "gang": e.gang,
                             "discovered": e.discovered, "country": e.country, "sector": e.sector,
                             "similarity": round(m[0], 2),
                             "claim_minus_start": (e.d - r.start).days if pd.notna(r.start) else None,
                             "submitted_minus_claim": (r.submitted - e.d).days if pd.notna(r.submitted) else None})
        h = pd.DataFrame(hits)
        out.append(h)
        sub = sub.assign(year=sub.ref.dt.year, hit=sub.name.isin(h.name) if len(h) else False)
        rep[kind] = {"organisations": len(sub), "matched": len(h),
                     "coverage_pct": round(len(h) / len(sub) * 100, 1) if len(sub) else None,
                     "coverage_by_year": {int(y): f"{int(g.hit.sum())}/{len(g)}" for y, g in sub.groupby("year")}}
        if len(h):
            rep[kind]["claim_minus_start_days"] = h.claim_minus_start.dropna().describe(percentiles=[.25, .5, .75]).round(1).to_dict()
            rep[kind]["submitted_minus_claim_days"] = h.submitted_minus_claim.dropna().describe(percentiles=[.25, .5, .75]).round(1).to_dict()
            rep[kind]["matched_country_US_pct"] = round((h.country == "US").mean() * 100, 1)
    pd.concat(out).assign(label="").to_csv(AUDIT / "wa_matches.csv", index=False)
    json.dump(rep, open(BUILD / "ext_wa.json", "w"), indent=2, default=str)
    print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main()
