#!/usr/bin/env python3
"""Valuta build/llm_validate_*.jsonl: accuratezza, copertura e soglia di confidenza.

Una predizione 'conta' se non e' unknown e non e' stata scartata per
evidenza da tracker. Si riportano:
  * copertura     = quota di eventi a cui il modello da' una risposta valida;
  * accuratezza   = quota corretta fra quelle date;
  * per tipo di input (solo nome / nome+dominio / nome+descrizione);
  * curva per soglia di confidenza, per scegliere dove accettare.
Riferimento per il settore: 63% di accordo fra le due fonti a parita' di
etichetta (tetto di qualita' delle etichette stesse).
"""
import json, sys
from pathlib import Path
import pandas as pd


def load(path, prompt_version=None):
    """Carica una corsa. Tiene solo la versione di prompt piu' recente (o
    quella indicata) e, per evento, l'ultimo tentativo."""
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    pv = prompt_version or max(r.get("prompt_version", "v0") for r in recs)
    rows = []
    for r in recs:
        if r.get("prompt_version") != pv:
            continue
        p = r.get("parsed") or {}
        rows.append({"event_id": r["event_id"], "input_type": r["input_type"], "true_c": r["true_country"], "true_s": r["true_sector"],
                     "pc": p.get("country"), "ps": p.get("sector"), "conf": p.get("confidence", 0.0),
                     "tracker": r.get("rejected") == "tracker_evidence", "error": "error" in r,
                     "n_search": r.get("n_searches", 0), "tok": (r.get("tokens_in") or 0) + (r.get("tokens_out") or 0)})
    df = pd.DataFrame(rows).drop_duplicates("event_id", keep="last").set_index("event_id")
    df.loc[df.tracker, ["pc", "ps"]] = None
    return df


def report(df, name):
    print(f"\n══ {name}  (n={len(df)}, errori {df.error.sum()}, scartati per tracker {df.tracker.sum()})")
    print(f"   ricerche medie {df.n_search.mean():.1f} | token medi per chiamata {df.tok.mean():.0f}")
    for attr, p, t in (("paese", "pc", "true_c"), ("settore", "ps", "true_s")):
        ans = df[df[p].notna()]
        print(f"   {attr:8} copertura {len(ans)/len(df)*100:5.1f}% | accuratezza {(ans[p]==ans[t]).mean()*100:5.1f}%")
        for it, g in df.groupby("input_type"):
            a = g[g[p].notna()]
            if len(a):
                print(f"      {it:12} n={len(g):3d} copertura {len(a)/len(g)*100:5.1f}% accuratezza {(a[p]==a[t]).mean()*100:5.1f}%")
    print("   soglia di confidenza  ->  copertura / accuratezza (paese | settore)")
    for th in (0.0, 0.5, 0.7, 0.8, 0.9):
        g = df[df.conf >= th]
        c = g[g.pc.notna()]; s = g[g.ps.notna()]
        print(f"      >= {th:.1f}   paese {len(c)/len(df)*100:5.1f}% / {(c.pc==c.true_c).mean()*100:5.1f}%"
              f"   settore {len(s)/len(df)*100:5.1f}% / {(s.ps==s.true_s).mean()*100:5.1f}%")


if __name__ == "__main__":
    for f in sorted(Path("build").glob("llm_validate_*.jsonl")):
        report(load(f), f.name)
