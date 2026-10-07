#!/usr/bin/env python3
"""Campioni per l'etichettatura manuale -> audit/label_*.csv

Tre file, ciascuno con due colonne da compilare (label_A, label_B: due
annotatori indipendenti, per calcolare l'accordo kappa) e una colonna note.

  label_A_merge_random.csv     100 eventi CASUALI con 2+ nomi diversi tra i record.
                               Stima il tasso reale di fusioni sbagliate.
                               Etichette: same / different / unsure
  label_B_duplicates.csv       100 coppie casuali tra le candidate a duplicato
                               perso (stessa gang, <= 7 giorni, nome simile o
                               stesso dominio). Etichette: same_attack /
                               same_victim_new_post / different / unsure
  label_C_imputed.csv          valori di paese e settore IMPUTATI, 30 per ogni
                               origine (*_src), stratificati. Si verifica sul web
                               (sito ufficiale, registro, LinkedIn; NON i tracker
                               ransomware). Etichette: correct / wrong / unsure;
                               se wrong, scrivere il valore giusto in note.
L'ordine delle righe e' casuale e le origini non sono mostrate nel file C, per
non condizionare chi etichetta: la chiave e' in audit/label_C_key.csv.
"""
import json, random
from pathlib import Path
import pandas as pd

BUILD, AUDIT = Path("build"), Path("audit")
SEED = 7


def main():
    AUDIT.mkdir(exist_ok=True)
    r = pd.read_json(BUILD / "record_to_event.json", dtype={"source_id": str}).drop_duplicates(["source", "source_id"])
    u = pd.read_json(BUILD / "unified.json", dtype={"source_id": str}).merge(r, on=["source", "source_id"])
    ev = pd.read_json("release/events_enriched.json")

    # A — eventi casuali con piu' nomi
    g = u.groupby("event_id").agg(gang=("gang", "first"),
                                  names=("victim", lambda s: list(dict.fromkeys(s))),
                                  domains=("domain", lambda s: list(dict.fromkeys(s.dropna()))),
                                  sources=("source", lambda s: sorted(set(s))))
    g = g[g.names.map(len) >= 2]
    a = g.sample(100, random_state=SEED).reset_index()
    a["names"] = a.names.map(lambda x: " || ".join(map(str, x[:6])))
    a["domains"] = a.domains.map(lambda x: " || ".join(x[:4]))
    a["sources"] = a.sources.map("+".join)
    a.assign(label_A="", label_B="", note="").to_csv(AUDIT / "label_A_merge_random.csv", index=False)

    # B — coppie candidate a duplicato perso (dal file dell'audit)
    b = pd.read_csv(AUDIT / "duplicate_candidates_sample.csv").drop(columns=["label"], errors="ignore")
    b.assign(label_A="", label_B="", note="").to_csv(AUDIT / "label_B_duplicates.csv", index=False)

    # C — valori imputati, 30 per origine
    rows = []
    for attr in ("country", "sector"):
        for src, grp in ev[~ev[f"{attr}_src"].isin(["observed", "missing"])].groupby(f"{attr}_src"):
            for _, e in grp.sample(min(30, len(grp)), random_state=SEED).iterrows():
                rows.append({"event_id": e.event_id, "attribute": attr, "value": e[attr], "src": src,
                             "victim": e.victim, "domain": e.domain, "discovered": e.discovered, "gang": e.gang})
    c = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    c.insert(0, "row", range(1, len(c) + 1))
    c[["row", "src"]].to_csv(AUDIT / "label_C_key.csv", index=False)
    c.drop(columns=["src"]).assign(label_A="", label_B="", note="").to_csv(AUDIT / "label_C_imputed.csv", index=False)

    print(f"A: {len(a)} eventi | B: {len(b)} coppie | C: {len(c)} valori "
          f"({c.groupby(['attribute', 'src']).size().to_dict()})")


if __name__ == "__main__":
    main()
