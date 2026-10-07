#!/usr/bin/env python3
"""Valuta site_sector.py validate: copertura e accuratezza per soglia di confidenza.

Due gruppi: 'agree' (fonti concordi sul settore, etichetta affidabile) e
'conflict' (fonti discordi: corretto se coincide con uno dei candidati).
Riferimento: l'accordo tra le fonti stesse sul settore e' ~60% (kappa 0.54),
quindi e' il tetto realistico per qualsiasi classificatore valutato su di esse.
"""
import json, sys
import pandas as pd

L = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "build/site_sector_validate.jsonl", encoding="utf-8")]
df = pd.DataFrame([r for r in L if "error" not in r])
df["pred"] = df.parsed.map(lambda p: (p or {}).get("sector"))
df["conf"] = df.parsed.map(lambda p: (p or {}).get("confidence") or 0)
df["ok_strict"] = df.pred == df.true_sector
df["ok_any"] = [p in (c or [t]) for p, c, t in zip(df.pred, df.sector_candidates, df.true_sector)]
df["group"] = df.sector_conflict.map({False: "agree", True: "conflict"})
print(f"eventi: {len(df)} (errori API: {len(L) - len(df)}) | token medi {df.tokens.mean():.0f}")
print(f"unknown: {df.pred.isna().mean()*100:.1f}%")
for g, d in [("tutti", df)] + list(df.groupby("group")):
    print(f"\n[{g}] n={len(d)}")
    for th in (0.0, 0.7, 0.8, 0.9, 0.95):
        a = d[d.pred.notna() & (d.conf >= th)]
        if len(a):
            print(f"  conf>={th:.2f}  copertura {len(a)/len(d)*100:5.1f}%  "
                  f"acc. stretta {a.ok_strict.mean()*100:5.1f}%  acc. su candidati {a.ok_any.mean()*100:5.1f}%")
ans = df[df.pred.notna() & ~df.sector_conflict]
print("\nconfusioni piu' frequenti (gruppo agree):")
print(ans[~ans.ok_strict].groupby(["true_sector", "pred"]).size().sort_values(ascending=False).head(10).to_string())
