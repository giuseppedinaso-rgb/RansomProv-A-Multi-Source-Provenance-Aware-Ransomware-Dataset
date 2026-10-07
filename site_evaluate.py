#!/usr/bin/env python3
"""Valuta build/site_validate.jsonl: precisione e copertura della regola del sito web.

Riporta:
  * quota di siti recuperati (snapshot archive.org, sito dal vivo, nessuno);
  * per ogni regola di decisione (strong / phone / phone_majority): copertura e
    precisione, cosi' da tenere solo le regole che reggono;
  * per ogni tipo di segnale (schema, vat, address, phone): precisione quando
    il segnale e' presente e univoco;
  * quota di snapshot successivi alla data dell'attacco (atteso zero: si chiede
    a archive.org lo snapshot precedente).
"""
import json
from collections import Counter
import pandas as pd

L = [json.loads(l) for l in open("build/site_validate.jsonl", encoding="utf-8")]
ev = pd.read_json("build/events.json").set_index("event_id")
df = pd.DataFrame(L)
n = len(df)
print(f"eventi validati: {n}")
print("origine:", Counter(str(o).split(":")[0] for o in df.origin.fillna("errore")).most_common())
print(f"pagina recuperata: {df.fetched.fillna(False).mean()*100:.1f}%")

arch = df[df.origin.fillna("").str.startswith("archive:")].copy()
arch["snap"] = pd.to_datetime(arch.origin.str[8:16], format="%Y%m%d")
arch["disc"] = pd.to_datetime(arch.event_id.map(ev.discovered))
print(f"snapshot successivi all'attacco: {(arch.snap > arch.disc).sum()} su {len(arch)}")

ans = df[df.country.notna()]
print(f"\nrisposta data: {len(ans)/n*100:.1f}% | precisione complessiva: {(ans.country==ans.true_country).mean()*100:.1f}%")
for rule, g in ans.groupby("rule"):
    print(f"  regola {rule:15} n={len(g):4d}  copertura {len(g)/n*100:5.1f}%  precisione {(g.country==g.true_country).mean()*100:5.1f}%")
ans_o = ans.origin.str.split(":").str[0]
for o, g in ans.groupby(ans_o):
    print(f"  origine {o:15} n={len(g):4d}  precisione {(g.country==g.true_country).mean()*100:5.1f}%")
print("motivi di non risposta:",df[df.country.isna()].rule.value_counts().to_dict())

print("\nprecisione per segnale (quando presente e univoco):")
for k in ("schema", "vat", "address", "phone"):
    v = df.signals.dropna().map(lambda s: s.get(k) or [])
    idx = v[v.map(lambda x: len(set(x)) == 1)].index
    if len(idx):
        pred = v[idx].map(lambda x: x[0]); true = df.loc[idx, "true_country"]
        print(f"  {k:8} n={len(idx):4d}  precisione {(pred==true).mean()*100:5.1f}%")
wrong = ans[ans.country != ans.true_country]
print("\nerrori:", Counter(zip(wrong.true_country, wrong.country)).most_common(10))
