#!/usr/bin/env python3
"""Stadio 3 — unificazione delle tre fonti in uno schema unico, un record per riga.

Nessuna fusione qui: ogni riga e' un record di una fonte, con i campi
normalizzati accanto ai grezzi. La fusione in eventi e' lo stadio 4.
"""
import json, re
import tldextract
from pathlib import Path
from collections import Counter
import pandas as pd

from normalize import to_iso2, gang_key, canonical_gang_map
from ransomlook_clean import clean_title, victim_key

RAW, BUILD = Path("raw"), Path("build"); BUILD.mkdir(exist_ok=True)


def dt(v):
    if v in (None, "", "null"):
        return pd.NaT
    x = pd.to_datetime(v, errors="coerce", format="ISO8601", utc=True)
    return x.tz_localize(None) if pd.notna(x) else pd.NaT


# lista dei suffissi pubblici inclusa nel pacchetto (nessun download): risultato
# riproducibile. Serve a ottenere il dominio registrabile: gov.br e' un suffisso,
# non un'organizzazione; acme.co.uk -> etichetta "acme".
_TLD = tldextract.TLDExtract(suffix_list_urls=())
DOMAIN_OK = re.compile(r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}")
# domini di terzi che alcune fonti riportano al posto del sito della vittima
THIRD_PARTY = {"zoominfo.com", "example.com", "dnb.com", "linkedin.com", "facebook.com",
               "google.com", "wikipedia.org", "crunchbase.com", "bloomberg.com"}


def domain_of(site):
    """Dominio della vittima, o None se il valore non e' un dominio utilizzabile:
    troncato (foo.co...), .onion, testo libero, aggregatore di terzi."""
    s = str(site or "").strip().lower()
    s = s.split("://")[-1].split("/")[0].split("?")[0].split(":")[0].removeprefix("www.")
    if not DOMAIN_OK.fullmatch(s) or s.endswith(".onion"):
        return None
    x = _TLD(s)
    if not x.domain or not x.suffix:
        return None
    return None if f"{x.domain}.{x.suffix}" in THIRD_PARTY else s


def domain_label(d):
    """Etichetta registrabile del dominio (acme.co.uk -> acme): base della chiave dkey."""
    return _TLD(d).domain if d else None


def s(v):
    v = (str(v).strip() if v is not None else "")
    return v or None


def main():
    rf = json.load(open(RAW / "ransomfeed_raw.json", encoding="utf-8"))["records"]
    rl = json.load(open(RAW / "ransomlive_raw.json", encoding="utf-8"))["records"]
    lk = json.load(open(RAW / "ransomlook_raw.json", encoding="utf-8"))["records"]
    print(f"in ingresso: ransomfeed {len(rf)} | ransomware.live {len(rl)} | ransomlook {len(lk)}")

    rows = []
    for r in rf:
        rows.append(dict(source="ransomfeed", source_id=r.get("id"),
                         victim_raw=s(r.get("victim")), victim=s(r.get("victim")),
                         domain=domain_of(r.get("website")) or domain_of(r.get("victim")),
                         gang_raw=s(r.get("gang")), discovered=dt(r.get("date")), attack=pd.NaT,
                         country_raw=s(r.get("country")), country_hint=None,
                         sector_raw=s(r.get("work_sector")), description=s(r.get("description"))))
    for r in rl:
        rows.append(dict(source="ransomlive", source_id=r.get("id"),
                         victim_raw=s(r.get("victim")), victim=s(r.get("victim")),
                         domain=domain_of(r.get("website")) or domain_of(r.get("victim")),
                         gang_raw=s(r.get("group")), discovered=dt(r.get("discovered")),
                         attack=dt(r.get("attackdate")),
                         country_raw=s(r.get("country")), country_hint=None,
                         sector_raw=s(r.get("activity")), description=s(r.get("description"))))
    n_cont = 0
    for p in lk:
        c = clean_title(p.get("post_title"))
        if c["continuation"]:
            n_cont += 1; continue
        rows.append(dict(source="ransomlook", source_id=p.get("misp_uuid"),
                         victim_raw=s(p.get("post_title")), victim=s(c["name"]),
                         domain=domain_of(c["domain"]), gang_raw=s(p.get("group_name")),
                         discovered=dt(p.get("discovered")), attack=pd.NaT,
                         country_raw=None, country_hint=c["cc"],
                         sector_raw=None, description=s(p.get("description"))))

    df = pd.DataFrame(rows)
    df["discovered"] = df["discovered"].fillna(df["attack"])

    # chiavi di identita' della vittima
    df["vkey"] = df["victim"].map(victim_key)
    df["dkey"] = df["domain"].map(lambda d: victim_key(domain_label(d)) if d else None)
    for k in ("vkey", "dkey"):
        df.loc[df[k].str.len() < 3, k] = None

    # gang canonica, costruita su tutte e tre le fonti insieme
    by_src = {src: df.loc[df["source"] == src, "gang_raw"].dropna().unique().tolist()
              for src in ("ransomfeed", "ransomlive", "ransomlook")}
    cmap, crule, review = canonical_gang_map(by_src)
    manual = Path("gang_aliases_manual.json")
    if manual.exists():
        man = json.load(open(manual, encoding="utf-8"))
        for var, can in man.get("merge", {}).items():
            for k, v in list(cmap.items()):
                if v == var:
                    cmap[k], crule[k] = can, "manual"
        skip = {tuple(sorted(p[:2])) for p in man.get("keep_separate", [])}
        done_pairs = {tuple(sorted((a, b))) for a, b in man.get("merge", {}).items()}
        review = [r for r in review if tuple(sorted(r[:2])) not in skip | done_pairs]
    df["gang"] = df["gang_raw"].map(lambda g: cmap.get(gang_key(g)) if g else None)

    # paese ISO dai valori osservati
    df["country"] = df["country_raw"].map(to_iso2)

    # settore nella tassonomia target (sector_crosswalk.json, da build_sector_crosswalk.py)
    cw = json.load(open("sector_crosswalk.json", encoding="utf-8"))["crosswalk"]
    unmapped = sorted({x for x in df["sector_raw"].dropna().unique() if x not in cw})
    if unmapped:
        raise SystemExit(f"etichette di settore non in crosswalk: {unmapped[:10]} "
                         "-> rilanciare build_sector_crosswalk.py")
    df["sector"] = df["sector_raw"].map(lambda x: cw[x]["target"] if x in cw else None)
    df["sector_map"] = df["sector_raw"].map(lambda x: cw[x]["method"] if x in cw else None)

    before = len(df)
    df = df[df["discovered"].notna() & (df["vkey"].notna() | df["dkey"].notna()) & df["gang"].notna()].copy()
    dropped = before - len(df)

    merged = sorted({(k, cmap[k], crule[k]) for k in cmap if cmap[k] != k})

    rep = {
        "input": {"ransomfeed": len(rf), "ransomlive": len(rl), "ransomlook": len(lk)},
        "ransomlook_continuations_dropped": n_cont,
        "dropped_no_date_or_identity_or_gang": dropped,
        "rows": len(df),
        "rows_by_source": df["source"].value_counts().to_dict(),
        "gang_raw_distinct": int(df["gang_raw"].nunique()),
        "gang_canonical_distinct": int(df["gang"].nunique()),
        "gang_aliases_merged": [f"{a} -> {b}  [{r}]" for a, b, r in merged],
        "gang_similar_for_review": [f"{a} ~ {b}  ({r})" for a, b, r in review],
        "country_observed_rows": int(df["country"].notna().sum()),
        "sector_observed_rows": int(df["sector"].notna().sum()),
        "sector_mapping_method_rows": df["sector_map"].value_counts().to_dict(),
    }
    df.drop(columns=["attack"]).to_json(BUILD / "unified.json", orient="records",
                                        date_format="iso", force_ascii=False)
    json.dump(rep, open(BUILD / "report_stage3.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"righe: {len(df)}  (scartate {dropped}, continuazioni RansomLook {n_cont})")
    print(df["source"].value_counts().to_string())
    print(f"gang: {rep['gang_raw_distinct']} nomi grezzi -> {rep['gang_canonical_distinct']} canoniche")
    print(f"alias fusi: {len(rep['gang_aliases_merged'])}")
    for a in rep["gang_aliases_merged"]:
        print("   ", a)
    print(f"coppie simili NON fuse, da rivedere a mano: {len(rep['gang_similar_for_review'])}")
    for a in rep["gang_similar_for_review"][:20]:
        print("   ", a)
    print("-> build/unified.json, build/report_stage3.json")


if __name__ == "__main__":
    main()
