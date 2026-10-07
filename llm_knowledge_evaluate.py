#!/usr/bin/env python3
"""Valuta llm_knowledge.py validate e sceglie le soglie -> llm_knowledge_policy.json.

Per ogni (attributo, tipo di input) si cerca la soglia di confidenza piu' bassa
la cui precisione raggiunge l'obiettivo con almeno MIN_ACCEPTED risposte.
Se nessuna soglia ci riesce, quella combinazione non si applica.

Obiettivi (gli stessi gia' fissati in llm_policy.py e documentati nel notebook):
  paese   90%  — sotto la precisione delle regole deterministiche migliori,
                  sopra inherited_victim (82%);
  settore 63%  — l'accordo tra le fonti sullo stesso evento e' ~60%: un'etichetta
                  imputata non deve essere meno affidabile di una osservata.
Il settore si valuta sugli eventi con fonti concordi (etichetta affidabile).
"""
import json, sys
import pandas as pd

TARGET = {"country": 0.90, "sector": 0.63}
MIN_ACCEPTED = 20
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]

L = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "build/llm_knowledge_validate.jsonl", encoding="utf-8")]
df = pd.DataFrame([r for r in L if "error" not in r and r.get("parsed")])
print(f"eventi: {len(df)} (errori o non parsabili: {len(L) - len(df)}) | token medi {df.tokens.mean():.0f}")
for attr in ("country", "sector"):
    df[f"p_{attr}"] = df.parsed.map(lambda p: p.get(attr))
    df[f"c_{attr}"] = df.parsed.map(lambda p: p.get(f"{attr}_confidence") or 0)

policy = {}
for attr in ("country", "sector"):
    print(f"\n=== {attr} (obiettivo {TARGET[attr]:.0%})")
    for it, g in df.groupby("itype"):
        if attr == "sector":
            g = g[~g.sector_conflict]
        print(f"  [{it}] n={len(g)}  unknown {g[f'p_{attr}'].isna().mean()*100:.0f}%")
        chosen = None
        for th in THRESHOLDS:
            a = g[g[f"p_{attr}"].notna() & (g[f"c_{attr}"] >= th)]
            if not len(a):
                continue
            acc = (a[f"p_{attr}"] == a[f"true_{attr}"]).mean()
            print(f"    conf>={th:.2f}  copertura {len(a)/len(g)*100:5.1f}%  precisione {acc*100:5.1f}%  (n={len(a)})")
            if chosen is None and acc >= TARGET[attr] and len(a) >= MIN_ACCEPTED:
                chosen = {"min_confidence": th, "precision": round(acc, 3),
                          "coverage": round(len(a) / len(g), 3), "n_validation": int(len(a))}
        policy[f"{attr}|{it}"] = chosen
        print(f"    -> {'soglia ' + str(chosen['min_confidence']) if chosen else 'NON applicato'}")

json.dump({"target": TARGET, "min_accepted": MIN_ACCEPTED, "rules": policy},
          open(sys.argv[2] if len(sys.argv) > 2 else "llm_knowledge_policy.json", "w"), indent=2)
print("\n-> llm_knowledge_policy.json")
