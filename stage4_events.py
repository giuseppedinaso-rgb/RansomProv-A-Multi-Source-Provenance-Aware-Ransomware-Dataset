#!/usr/bin/env python3
"""Stadio 4 — dai record agli eventi, con provenienza degli attributi.

Un EVENTO e' un attacco rivendicato: stessa vittima, stessa gang, data vicina.

1. Identita' della vittima. Due record appartengono alla stessa vittima se,
   a parita' di gang, condividono la chiave del nome o quella del dominio
   (union-find: 'Acme Inc' e 'acme.com' si legano se un record li porta
   entrambi).
2. Eventi. Dentro ciascuna coppia vittima-gang, i record di ransomfeed e
   ransomware.live si raggruppano per data: record distanti al piu'
   WINDOW giorni sono lo stesso evento.
3. RansomLook. Le sue date di scoperta sono ricostruite a posteriori per il
   periodo precedente al 2022 (scarti mediani di mesi rispetto alle altre
   fonti), quindi NON le usiamo per separare eventi: un post di RansomLook
   la cui coppia vittima-gang esiste gia' nelle fonti ricche corrobora
   l'evento piu' vicino nel tempo. Solo i post senza alcun riscontro
   generano eventi propri.
4. Attributi. Paese e settore si prendono dai record dell'evento
   (country_src/sector_src = 'observed'). Se mancano, si ereditano da altri
   eventi della stessa vittima con qualunque gang ('inherited_victim': paese
   e settore descrivono la vittima, non l'attacco). Il paese si puo' infine
   dedurre dal titolo RansomLook ('title_hint') o dal ccTLD ('cctld').
"""
import json, os
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd

from normalize import cctld_to_iso2, mentions_country, single_country_mention

BUILD = Path("build")
WINDOW = int(os.environ.get("EVENT_WINDOW", 7))     # giorni (v1.0: 3; v1.1: 7 dopo l'audit dei duplicati)
OUT = Path(os.environ.get("EVENT_OUT", BUILD))      # cartella di uscita (default: build)
PLACEHOLDER_MIN_DOMAINS = 3

RICH = ("ransomfeed", "ransomlive")


class UF:
    def __init__(self): self.p = {}
    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


def resolve_country(rich):
    """Paese di un evento dai record delle fonti ricche.

    Se le fonti concordano: 'unanimous'. Se no, in ordine:
      1. 'cctld'       - il ccTLD geografico del dominio e' uno dei candidati;
      2. 'description' - la descrizione nomina esattamente uno dei candidati;
      3. 'source_priority' - vince ransomfeed. Misurato sui conflitti:
         arbitrati dal ccTLD ransomfeed ha ragione 316/410 volte contro 116/414
         di ransomware.live; fuori campione, arbitrati solo dalla descrizione,
         258/419 contro 170/420.
    """
    cands = sorted(set(rich.country.dropna()))
    if not cands:
        return None, None, []
    if len(cands) == 1:
        return cands[0], "unanimous", cands
    tld = next((t for t in rich.domain.map(cctld_to_iso2) if t), None)
    if tld in cands:
        return tld, "cctld", cands
    text = " ".join(str(d) for d in rich.description if d)
    hits = [c for c in cands if mentions_country(text, c)]
    if len(hits) == 1:
        return hits[0], "description", cands
    rf = rich[rich.source == "ransomfeed"].country.dropna()
    pool = rf if len(rf) else rich.country.dropna()
    return Counter(pool).most_common(1)[0][0], "source_priority", cands


def resolve_sector(rich):
    """Settore di un evento. Se le fonti concordano: 'unanimous'. Se no vince
    ransomware.live, perche' la sua etichetta e' nativa della tassonomia
    target e non passa per il crosswalk ('source_priority')."""
    cands = sorted(set(rich.sector.dropna()))
    if not cands:
        return None, None, []
    if len(cands) == 1:
        return cands[0], "unanimous", cands
    rl = rich[rich.source == "ransomlive"].sector.dropna()
    pool = rl if len(rl) else rich.sector.dropna()
    return Counter(pool).most_common(1)[0][0], "source_priority", cands


def majority(values):
    c = Counter(v for v in values if v)
    if not c:
        return None, 0
    (top, n), *rest = c.most_common()
    return top, len(c)


def main():
    df = pd.read_json(BUILD / "unified.json", orient="records", convert_dates=["discovered"])
    df["discovered"] = pd.to_datetime(df["discovered"], errors="coerce")
    print(f"record: {len(df)}")

    # --- 1. identita' vittima per gang ---------------------------------
    # nomi segnaposto: alcune gang pubblicano "FILES", "Breach", "Full data dump"
    # al posto del nome della vittima. Un nome che nella stessa gang compare con
    # PLACEHOLDER_MIN_DOMAINS o piu' domini diversi non identifica una vittima:
    # non viene usato per collegare record (ne' per l'ereditarieta').
    for c in ("vkey", "dkey"):
        df[c] = df[c].where(df[c].notna(), None)
    nd = df.dropna(subset=["vkey", "dkey"]).groupby(["gang", "vkey"]).dkey.nunique()
    placeholder = set(nd[nd >= PLACEHOLDER_MIN_DOMAINS].index)
    df["vkey"] = [None if (g, vk) in placeholder else vk for g, vk in zip(df.gang, df.vkey)]
    print(f"nomi segnaposto scartati: {len(placeholder)}")
    # record rimasti senza alcuna chiave: ciascuno resta un'identita' a se'
    df["vkey"] = [vk if (vk or dk) else f"_rec{i}" for i, (vk, dk) in enumerate(zip(df.vkey, df.dkey))]

    uf = UF()
    # v1.1: nomi troncati dal leak site ("roemerlueftu" / "roemerlueftung"): nella
    # stessa gang si collegano le chiavi di cui una e' l'inizio dell'altra
    # (almeno 8 caratteri, rapporto di lunghezza >= 0.6)
    for g, grp in df.groupby("gang"):
        ks = sorted({k for k in grp.vkey if k and not str(k).startswith("_rec") and len(k) >= 8})
        for a, b in zip(ks, ks[1:]):
            if b.startswith(a) and len(a) / len(b) >= 0.6:
                uf.union(f"{g}|v|{a}", f"{g}|v|{b}")
    for g, vk, dk in zip(df.gang, df.vkey, df.dkey):
        toks = [f"{g}|v|{vk}" if vk else None, f"{g}|d|{dk}" if dk else None]
        toks = [t for t in toks if t]
        for t in toks: uf.find(t)
        if len(toks) == 2: uf.union(toks[0], toks[1])
    df["vid"] = [uf.find(f"{g}|v|{vk}") if vk else uf.find(f"{g}|d|{dk}")
                 for g, vk, dk in zip(df.gang, df.vkey, df.dkey)]

    # --- 2+3. eventi ----------------------------------------------------
    event_of = {}
    eid = 0
    for vid, grp in df.groupby("vid", sort=False):
        rich = grp[grp.source.isin(RICH)].sort_values("discovered")
        look = grp[grp.source == "ransomlook"].sort_values("discovered")
        anchors = []                                   # (eid, data rappresentativa)
        last = None
        for i, d in zip(rich.index, rich.discovered):
            if last is None or (d - last).days > WINDOW:
                eid += 1; anchors.append((eid, d))
            event_of[i] = eid; last = d
        if anchors:                                    # RansomLook corrobora
            for i, d in zip(look.index, look.discovered):
                event_of[i] = min(anchors, key=lambda a: abs((a[1] - d).days))[0]
        else:                                          # solo RansomLook
            last = None
            for i, d in zip(look.index, look.discovered):
                if last is None or (d - last).days > WINDOW:
                    eid += 1
                event_of[i] = eid; last = d
    df["event_id"] = df.index.map(event_of)

    # --- 4a. attributi osservati ---------------------------------------
    rows, conflicts_c, conflicts_s = [], 0, 0
    for e, grp in df.groupby("event_id", sort=True):
        srcs = sorted(grp.source.unique())
        rich = grp[grp.source.isin(RICH)]
        base = rich if len(rich) else grp
        c, how, cands = resolve_country(rich)
        nc = len(cands)
        sct, show, scands = resolve_sector(rich)
        ns = len(scands)
        conflicts_c += nc > 1; conflicts_s += ns > 1
        names = [v for v in base.victim if v]
        rows.append({
            "event_id": int(e),
            "discovered": base.discovered.min().date().isoformat(),
            "victim": max(names, key=len) if names else None,
            "domain": next((d for d in base.domain if d), None),
            "gang": base.gang.iloc[0],
            "gang_raw": Counter(base.gang_raw).most_common(1)[0][0],
            "country": c, "country_src": "observed" if c else None,
            "country_resolution": how, "country_candidates": cands if len(cands) > 1 else None,
            "sector": sct, "sector_src": "observed" if sct else None,
            "sector_resolution": show, "sector_candidates": scands if len(scands) > 1 else None,
            "sources": srcs, "n_sources": len(srcs),
            "country_conflict": nc > 1, "sector_conflict": ns > 1,
            "sectors_raw": sorted({x for x in rich.sector_raw.dropna()}),
            "_vkeys": sorted({k for k in grp.vkey if k and not k.startswith("_rec")}), "_dkeys": sorted({k for k in grp.dkey if k}),
            "_hint": next((h for h in grp.country_hint if h), None),
            "_desc": " ".join(str(d) for d in grp.description if d),
        })
    ev = pd.DataFrame(rows)

    # --- 4b. ereditarieta' per vittima (qualunque gang) -----------------
    vc, vs = defaultdict(Counter), defaultdict(Counter)
    for _, r in ev[ev.country_src == "observed"].iterrows():
        for k in r._vkeys + r._dkeys: vc[k][r.country] += 1
    for _, r in ev[ev.sector_src == "observed"].iterrows():
        for k in r._vkeys + r._dkeys: vs[k][r.sector] += 1

    def inherit(keys, table):
        c = Counter()
        for k in keys: c.update(table.get(k, {}))
        return c.most_common(1)[0][0] if c else None

    for i, r in ev.iterrows():
        keys = r._vkeys + r._dkeys
        if not r.country:
            v = inherit(keys, vc)
            if v:
                ev.at[i, "country"], ev.at[i, "country_src"] = v, "inherited_victim"
            # title_hint (paese dal titolo del post) non e' piu' applicato:
            # precisione 69.8% (n=63) in validate_rules.py, sotto le altre regole
            elif cctld_to_iso2(r.domain):
                ev.at[i, "country"], ev.at[i, "country_src"] = cctld_to_iso2(r.domain), "cctld"
            elif single_country_mention(r._desc):
                ev.at[i, "country"], ev.at[i, "country_src"] = single_country_mention(r._desc), "description"
        if not r.sector:
            v = inherit(keys, vs)
            if v:
                ev.at[i, "sector"], ev.at[i, "sector_src"] = v, "inherited_victim"

    # chiavi e indizi per evento: servono a validate_rules.py per misurare la
    # precisione di ogni regola di inferenza su eventi con valore osservato
    ev[["event_id", "_vkeys", "_dkeys", "_hint", "_desc"]].to_json(
        OUT / "event_keys.json", orient="records", force_ascii=False)
    ev = ev.drop(columns=["_vkeys", "_dkeys", "_hint", "_desc"])
    ev["country_src"] = ev["country_src"].fillna("missing")
    ev["sector_src"] = ev["sector_src"].fillna("missing")

    # --- report ----------------------------------------------------------
    combo = ev.sources.map(lambda s: "+".join(s)).value_counts().to_dict()
    rep = {
        "window_days": WINDOW, "records": len(df), "events": len(ev),
        "events_by_n_sources": ev.n_sources.value_counts().sort_index().to_dict(),
        "events_by_source_combination": combo,
        "country_src": ev.country_src.value_counts().to_dict(),
        "sector_src": ev.sector_src.value_counts().to_dict(),
        "country_conflicts_observed": int(conflicts_c),
        "country_resolution": ev.country_resolution.value_counts(dropna=True).to_dict(),
        "sector_conflicts_observed": int(conflicts_s),
        "sector_resolution": ev.sector_resolution.value_counts(dropna=True).to_dict(),
        "sector_distribution": ev.sector.value_counts(dropna=False).to_dict(),
        "victims_distinct": int(ev.victim.nunique()),
        "gangs_distinct": int(ev.gang.nunique()),
        "countries_distinct": int(ev.country.nunique()),
        "period": [ev.discovered.min(), ev.discovered.max()],
    }
    ev.to_json(OUT / "events.json", orient="records", force_ascii=False, indent=1)
    json.dump(rep, open(OUT / "report_stage4.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    df[["source", "source_id", "event_id"]].to_json(OUT / "record_to_event.json", orient="records")

    print(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    print("-> build/events.json, build/record_to_event.json, build/report_stage4.json")


if __name__ == "__main__":
    main()
