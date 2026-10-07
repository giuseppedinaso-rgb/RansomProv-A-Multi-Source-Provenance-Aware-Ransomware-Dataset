#!/usr/bin/env python3
"""Dalla validazione alla policy di applicazione -> llm_policy.json.

Per ogni tipo di input (solo nome / nome+dominio / nome+descrizione) confronta
il modello CON e SENZA ricerca sugli stessi eventi. La ricerca si usa solo
dove alza l'accuratezza di almeno MIN_GAIN punti su almeno un attributo:
altrove costa ~20 volte i token senza beneficio misurato.

Per ogni attributo e modalita' fissa poi la SOGLIA DI CONFIDENZA minima tale
che l'accuratezza delle risposte accettate sia >= TARGET_ACC. Le risposte
sotto soglia restano 'missing'.

Obiettivi:
  paese   TARGET_ACC_COUNTRY = 0.90  (le regole deterministiche arrivano al 97%)
  settore TARGET_ACC_SECTOR  = 0.63  (accordo fra le due fonti a parita' di etichetta:
                                      oltre questo non si puo' misurare)
"""
import json
from pathlib import Path
import pandas as pd
from llm_evaluate import load

MIN_GAIN = 5.0
TARGET_ACC_COUNTRY, TARGET_ACC_SECTOR = 0.90, 0.63
MIN_ACCEPTED = 20           # soglie calcolate su almeno 20 risposte accettate


def acc(df, p, t):
    a = df[df[p].notna()]
    return (a[p] == a[t]).mean() * 100 if len(a) else float("nan"), len(a) / len(df) * 100 if len(df) else 0


def threshold(df, p, t, target):
    for th in [x / 100 for x in range(0, 100, 5)]:
        a = df[(df.conf >= th) & df[p].notna()]
        if len(a) >= MIN_ACCEPTED and (a[p] == a[t]).mean() >= target:
            return th, round((a[p] == a[t]).mean(), 3), round(len(a) / len(df), 3)
    return None, None, None                  # nessuna soglia raggiunge l'obiettivo


def main():
    S = load("build/llm_validate_search.jsonl")
    N = load("build/llm_validate_nosearch.jsonl")
    common = set(S.index) & set(N.index)
    print(f"eventi validati: con ricerca {len(S)} | senza {len(N)}")

    search_for, detail = [], {}
    print(f"\n{'input':12} {'attr':8} {'con ricerca':>22} {'senza ricerca':>22}  guadagno")
    for it in sorted(set(S.input_type) | set(N.input_type)):
        s, n = S[S.input_type == it], N[N.input_type == it]
        gains = {}
        for attr, p, t in (("paese", "pc", "true_c"), ("settore", "ps", "true_s")):
            (as_, cs), (an, cn) = acc(s, p, t), acc(n, p, t)
            gains[attr] = as_ - an
            print(f"{it:12} {attr:8} acc {as_:5.1f}% cop {cs:5.1f}%   acc {an:5.1f}% cop {cn:5.1f}%   {as_-an:+5.1f}")
        use = any(g >= MIN_GAIN for g in gains.values() if g == g)
        if use:
            search_for.append(it)
        detail[it] = {"gain_country": round(gains["paese"], 1), "gain_sector": round(gains["settore"], 1),
                      "use_search": use}

    thr = {}
    for mode, df in (("search", S), ("nosearch", N)):
        for attr, p, t, tgt in (("country", "pc", "true_c", TARGET_ACC_COUNTRY),
                                ("sector", "ps", "true_s", TARGET_ACC_SECTOR)):
            th, a, c = threshold(df, p, t, tgt)
            thr[f"{mode}_{attr}"] = {"min_confidence": th, "accuracy": a, "coverage": c, "target": tgt}

    pol = {"search_for_input_types": search_for, "min_gain_points": MIN_GAIN,
           "per_input_type": detail, "thresholds": thr}
    json.dump(pol, open("llm_policy.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nricerca per: {search_for}")
    for k, v in thr.items():
        print(f"  {k:18} soglia {v['min_confidence']}  accuratezza {v['accuracy']}  copertura {v['coverage']}")
    print("-> llm_policy.json")


if __name__ == "__main__":
    main()
