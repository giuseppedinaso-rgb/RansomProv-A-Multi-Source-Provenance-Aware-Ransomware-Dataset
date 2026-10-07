#!/usr/bin/env python3
"""Costruisce sector_crosswalk.json: etichetta di settore grezza -> classe target.

TASSONOMIA TARGET: quella di ransomware.live (12 classi + Other), per tre
ragioni: e' esterna e gia' in uso, copre la fonte piu' ricca, e le sue
etichette entrano senza conversione.

MAPPATURA DELLE ETICHETTE DI RANSOMFEED (175), in due modi:
  * 'empirical': sugli eventi etichettati da entrambe le fonti si prende la
    classe ransomware.live piu' frequente, se supporto >= MIN_N e
    purezza >= MIN_PURITY;
  * 'manual': tutte le altre, decise a mano sotto (MANUAL), con il
    crosstab come riferimento e la convenzione di ransomware.live dove il
    segnale c'e' (Construction -> Manufacturing, 44%).
"""
import json
from pathlib import Path
import pandas as pd

MIN_N, MIN_PURITY = 20, 0.5
TARGET = ["Professional Services", "Manufacturing", "Technology", "Healthcare",
          "Retail & E-Commerce", "Financial Services", "Transportation", "Education",
          "Government & Defense", "Agriculture and Food Production", "Energy & Utilities",
          "Hospitality", "Other"]
MISSING = {"Not Found", "Unknown", "Unknown/Not Found", "N/A", "", None}

PS, MF, TE, HC, RE, FI, TR, ED, GD, AG, EN, HO, OT = TARGET
MANUAL = {
    # alto volume, segnale empirico debole: convenzione ransomware.live o semantica
    "Construction": MF, "Services": PS, "Retail (distribution)": RE, "Automotive": MF,
    "Engineering consulting": PS, "Real Estate / Real Estate Management": PS,
    "Distribution": RE, "Medias and audiovisual": TE, "Human Services": HC,
    "Entertainment industry": HO, "Sports": HO, "Electricity": EN, "Utility": EN,
    "Non-Governmental Organizations (NGOs)": OT, "Shops": RE, "Research": PS,
    "Personal care": RE, "Culture and entertainment": HO, "Luxury": RE,
    "Lodging industry": HO, "Mining": MF, "Church": OT, "Civil society": OT,
    "Fintech": FI, "Charity associations": OT, "Market infrastructures": FI,
    "Culture": OT, "Avionics": MF, "Defense industry": MF, "Defense": GD,
    "Water distribution and supply": EN, "Defense ministries (including the military)": GD,
    "International organizations": GD, "Development": TE, "Critical Industries": MF,
    "Judicial power (justice)": GD, "Political parties": GD, "Think tanks": PS,
    "Flight": TR, "Defense research and development": MF,
    # basso volume: semantica
    "Real Estate": PS, "Ministries of foreign affairs": GD, "Software Development": TE,
    "Financial Services": FI, "Critical Sectors": OT, "Retail": RE, "Publishing": TE,
    "Nuclear power (civilian use)": EN, "Insurance": FI, "Oil and Gas": EN,
    "Information Technology": TE, "Information Technology and Services": TE,
    "Embassy": GD, "Wholesale and Retail Trade": RE, "Food and Beverage Services": HO,
    "Machinery Manufacturing": MF, "Food and Beverage Manufacturing": AG,
    "Food & Beverages": AG, "Pharmaceutical Distribution": HC, "Building Materials": MF,
    "Non-Destructive Testing and Inspection Services": PS,
    "Marine Equipment Manufacturing": MF, "Retail Groceries": RE,
    "Plastic Processing and Molding": MF, "Engineering and Manufacturing": MF,
    "Manufacturing and Geotechnical Solutions": MF, "Advertising Services": PS,
    "Information and Communications Technology (ICT) Solutions and Services": TE,
    "Real Estate Investment and Mergers & Acquisitions": FI,
    "Film and Video Production": TE, "Facilities Services": PS, "Audiovisual Services": TE,
    "Financial Technology": FI, "Computer Hardware Manufacturing": MF,
    "Freight Brokerage": TR, "Metal and Mineral Merchant Wholesalers": RE,
    "Automation Machinery Manufacturing": MF, "Property Management and Real Estate": PS,
    "Real Estate Management": PS, "Legal Services": PS, "Diplomacy": GD,
    "Audio equipment manufacturing": MF, "Printing Services": PS, "Healthcare Management": HC,
    "Media and Communication": TE, "IT Services": TE, "Access Control Solutions": TE,
    "Food": AG, "Energy Services": EN, "IT Services and IT Consulting": TE,
    "Research and Development": PS, "E-commerce and Financial Technology": RE,
    "Printing Equipment and Supplies": MF,
    "Semiconductor and Electronic Component Manufacturing": MF, "Automotive Retail": RE,
    "Precision Engineering": MF, "Food Production": AG, "Building Cleaning Services": PS,
    "Employee Services": PS, "Collective Management Organization": PS,
    "Freight & Logistics Services": TR, "Non-profit Organization": OT,
    "Healthcare Storage Solutions": HC, "Ambulance Services": HC, "Private Equity": FI,
    "Maritime Electronics and Naval Systems": MF, "Electrical Engineering and IT Services": TE,
    "Performing Arts": HO, "Diversified Conglomerate": OT, "Precision Manufacturing": MF,
    "Surface Technology": MF, "Occupational Health and Employment Testing": HC,
    "Biotechnology": HC, "Housing and Financial Services": FI,
    "Logistics and Freight Forwarding": TR,
    "Engineering, Procurement, and Construction Management (EPCM)": PS, "Healthcare": HC,
    "Business Services": PS, "Training": ED,
}


def main():
    u = pd.read_json("build/unified.json")
    m = pd.read_json("build/record_to_event.json"); u["event_id"] = m.event_id.values
    rf = u[(u.source == "ransomfeed") & u.sector_raw.notna()][["event_id", "sector_raw"]]
    rl = u[(u.source == "ransomlive") & u.sector_raw.isin(TARGET)][["event_id", "sector_raw"]]
    x = rf.merge(rl, on="event_id", suffixes=("_rf", "_rl")).drop_duplicates()
    ct = pd.crosstab(x.sector_raw_rf, x.sector_raw_rl)

    cw = {}
    for lab in TARGET:                                   # etichette native
        cw[lab] = {"target": lab, "method": "native"}
    for lab in MISSING:
        if lab:
            cw[lab] = {"target": None, "method": "missing"}

    unresolved = []
    for lab in u[u.source == "ransomfeed"].sector_raw.dropna().unique():
        if lab in MISSING or lab in TARGET:      # mancanti o gia' nativi
            continue
        n = int(ct.loc[lab].sum()) if lab in ct.index else 0
        top = ct.loc[lab].idxmax() if n else None
        pur = float(ct.loc[lab].max() / n) if n else 0.0
        if lab in MANUAL:
            cw[lab] = {"target": MANUAL[lab], "method": "manual", "n": n,
                       "empirical": top, "purity": round(pur, 3)}
        elif n >= MIN_N and pur >= MIN_PURITY:
            cw[lab] = {"target": top, "method": "empirical", "n": n, "purity": round(pur, 3)}
        else:
            unresolved.append((lab, n, round(pur, 2), top))
    if unresolved:
        for r in unresolved:
            print("  senza regola:", r)
        raise SystemExit(f"{len(unresolved)} etichette senza regola -> aggiungerle a MANUAL")

    bad = {k: v["target"] for k, v in cw.items() if v["target"] not in TARGET + [None]}
    assert not bad, f"target fuori tassonomia: {bad}"
    json.dump({"target_taxonomy": TARGET, "min_n": MIN_N, "min_purity": MIN_PURITY,
               "crosswalk": cw}, open("sector_crosswalk.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    from collections import Counter
    print("metodi:", Counter(v["method"] for v in cw.values()))
    print("-> sector_crosswalk.json")


if __name__ == "__main__":
    main()
