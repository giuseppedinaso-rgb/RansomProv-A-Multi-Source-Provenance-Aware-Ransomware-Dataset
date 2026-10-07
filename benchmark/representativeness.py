#!/usr/bin/env python3
"""Test 2 — rappresentativita' delle rivendicazioni rispetto alle vittime confermate.

Riferimento: notifiche di violazione dello Stato di Washington (data.wa.gov,
sb4j-ca4h) con CyberattackType = Ransomware, giugno 2020 - giugno 2026
(build/ext_wa_raw.csv) e abbinamenti con il dataset (audit/wa_matches.csv).
Le 13 classi del dataset sono ricondotte alle 6 categorie di Washington:
  Healthcare -> Health; Financial Services -> Finance; Education -> Education;
  Government & Defense -> Government; Other -> Non-Profit/Charity;
  tutte le altre -> Business.
(Other contiene anche media e sport: la corrispondenza con Non-Profit e'
approssimata.)

Misure:
  1. distribuzione per categoria: rivendicazioni con vittima USA (stesso periodo)
     contro vittime ransomware confermate; rapporto di rappresentazione e
     test chi-quadro;
  2. copertura: quota di vittime confermate che compaiono nel dataset, per
     categoria e per numero di residenti coinvolti (intervalli di confidenza di
     Wilson);
  3. accordo tra il settore del dataset e la categoria ufficiale sugli
     abbinamenti (controllo esterno delle etichette di settore).
Uscita: benchmark/results/representativeness.json
"""
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parents[1]
MAP = {"Healthcare": "Health", "Financial Services": "Finance", "Education": "Education",
       "Government & Defense": "Government", "Other": "Non-Profit/Charity"}
CATS = ["Business", "Health", "Finance", "Education", "Non-Profit/Charity", "Government"]


def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round((c - h) * 100, 1), round((c + h) * 100, 1)]


def main():
    wa = pd.read_csv(ROOT / "build" / "ext_wa_raw.csv", parse_dates=["ref"])
    rw = wa[wa.cyberattacktype == "Ransomware"].drop_duplicates("name")
    m = pd.read_csv(ROOT / "audit" / "wa_matches.csv")
    m = m[(m.kind == "ransomware") & (m.get("review_assistant", "correct") != "wrong")]
    ev = pd.read_json(ROOT / "release" / "events_enriched.json")
    ev["d"] = pd.to_datetime(ev.discovered)
    us = ev[(ev.country == "US") & ev.sector.notna() & (ev.d >= "2020-06-01") & (ev.d <= "2026-06-30")]
    us_cat = us.sector.map(lambda s: MAP.get(s, "Business"))

    rep = {}
    a = rw.industrytype.value_counts().reindex(CATS, fill_value=0)
    b = us_cat.value_counts().reindex(CATS, fill_value=0)
    chi2, p, _, _ = chi2_contingency(np.vstack([a.values, b.values]))
    rep["distribution"] = {c: {"confirmed_pct": round(a[c] / a.sum() * 100, 1), "claims_US_pct": round(b[c] / b.sum() * 100, 1),
                               "ratio_claims_to_confirmed": round((b[c] / b.sum()) / max(a[c] / a.sum(), 1e-9), 2)} for c in CATS}
    rep["distribution_test"] = {"chi2": round(chi2, 1), "p_value": float(p), "n_confirmed": int(a.sum()), "n_claims_US": int(b.sum())}

    rw = rw.assign(hit=rw.name.isin(m.name))
    rep["coverage_by_industry"] = {c: {"confirmed": int((g := rw[rw.industrytype == c]).shape[0]), "on_leak_site": int(g.hit.sum()),
                                       "pct": round(g.hit.mean() * 100, 1) if len(g) else None, "ci95": wilson(int(g.hit.sum()), len(g))}
                                   for c in CATS}
    size = pd.to_numeric(rw.washingtoniansaffected, errors="coerce")
    bins = pd.cut(size, [0, 1000, 5000, 20000, 1e12], labels=["500-1k", "1k-5k", "5k-20k", ">20k"], right=True)
    rep["coverage_by_wa_residents_affected"] = {str(k): {"confirmed": len(g), "on_leak_site": int(g.hit.sum()),
                                                         "pct": round(g.hit.mean() * 100, 1), "ci95": wilson(int(g.hit.sum()), len(g))}
                                                for k, g in rw.groupby(bins, observed=True)}

    mm = m.merge(rw[["name", "industrytype"]], on="name").dropna(subset=["sector"])
    mm["ours"] = mm.sector.map(lambda s: MAP.get(s, "Business"))
    rep["sector_agreement_on_matches"] = {"n": len(mm), "agreement_pct": round((mm.ours == mm.industrytype).mean() * 100, 1),
                                          "ci95": wilson(int((mm.ours == mm.industrytype).sum()), len(mm)),
                                          "confusions": mm[mm.ours != mm.industrytype].groupby(["industrytype", "ours"]).size().to_dict()}
    rep["sector_agreement_on_matches"]["confusions"] = {f"{k[0]} -> {k[1]}": v for k, v in rep["sector_agreement_on_matches"]["confusions"].items()}
    json.dump(rep, open(ROOT / "benchmark" / "results" / "representativeness.json", "w"), indent=2, default=str)
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
