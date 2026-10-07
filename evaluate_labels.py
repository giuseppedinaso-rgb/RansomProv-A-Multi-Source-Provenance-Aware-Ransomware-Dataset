#!/usr/bin/env python3
"""Risultati dell'etichettatura manuale (audit/label_*.csv) -> build/report_labels.json.

Per ogni file: distribuzione delle etichette (annotatore A, o consenso se ci
sono due annotatori e concordano), tasso di errore con intervallo di Wilson al
95% (le righe 'unsure' sono escluse dal denominatore e contate a parte), e
kappa di Cohen tra annotatori quando entrambe le colonne sono compilate.
"""
import json, math, sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import cohen_kappa_score

AUDIT = Path("audit")


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round((c - h) * 100, 1), round((c + h) * 100, 1)]


COL_A, COL_B = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("label_A", "label_B")


def labels(df):
    """Colonne da usare: label_A/label_B (umane, default) oppure, da riga di comando,
    es. `python3 evaluate_labels.py label_ai label_A` per l'annotazione AI con
    l'accordo rispetto a un annotatore umano."""
    a = df[COL_A].fillna("").astype(str).str.strip().str.lower() if COL_A in df else pd.Series("", index=df.index)
    b = df[COL_B].fillna("").astype(str).str.strip().str.lower() if COL_B in df else pd.Series("", index=df.index)
    both = (a != "") & (b != "")
    kappa = round(cohen_kappa_score(a[both], b[both]), 3) if both.sum() >= 10 else None
    final = a.where(~both | (a == b), "unsure")          # disaccordo -> unsure
    return final, kappa, int(both.sum())


def rate(final, bad, good):
    n = int(final.isin(bad + good).sum()); k = int(final.isin(bad).sum())
    return {"n_decided": n, "errors": k, "error_pct": round(k / n * 100, 1) if n else None,
            "ci95": wilson(k, n), "unsure": int((final == "unsure").sum())}


def main():
    rep = {}
    f = AUDIT / "label_A_merge_random.csv"
    if f.exists():
        df = pd.read_csv(f); fin, k, nb = labels(df)
        if (fin != "").any():
            rep["A_over_merge"] = {**rate(fin, ["different"], ["same"]), "kappa": k, "n_double": nb}
    f = AUDIT / "label_B_duplicates.csv"
    if f.exists():
        df = pd.read_csv(f); fin, k, nb = labels(df)
        if (fin != "").any():
            rep["B_missed_duplicates_among_candidates"] = {
                **rate(fin, ["same_attack"], ["same_victim_new_post", "different"]),
                "distribution": fin[fin != ""].value_counts().to_dict(), "kappa": k, "n_double": nb}
    f = AUDIT / "label_C_imputed.csv"
    if f.exists():
        df = pd.read_csv(f).merge(pd.read_csv(AUDIT / "label_C_key.csv"), on="row")
        fin, k, nb = labels(df); df["final"] = fin
        if (fin != "").any():
            rep["C_imputed"] = {"kappa": k, "n_double": nb, "by_src": {
                f"{a}|{s}": rate(g.final, ["wrong"], ["correct"]) for (a, s), g in df.groupby(["attribute", "src"])}}
    rep["columns"] = [COL_A, COL_B]
    json.dump(rep, open(f"build/report_labels_{COL_A}.json", "w"), indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
