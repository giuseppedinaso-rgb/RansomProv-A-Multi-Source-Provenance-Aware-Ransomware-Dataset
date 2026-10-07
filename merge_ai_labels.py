#!/usr/bin/env python3
"""Inserisce le etichette dell'annotazione assistita da LLM (audit/ai_labels/*.json)
nelle colonne label_ai / note_ai dei file audit/label_*.csv. Le colonne umane
(label_A, label_B) non vengono toccate.
Provenienza: agenti Claude (Anthropic) con ricerca web; i tracker ransomware erano
vietati come fonte di evidenza; per il file C l'origine del valore era nascosta."""
import json, glob
from pathlib import Path
import pandas as pd

A = Path("audit")
a = pd.read_csv(A / "label_A_merge_random.csv")
if (A / "ai_labels/A.json").exists():
    j = pd.DataFrame(json.load(open(A / "ai_labels/A.json"))).set_index("event_id")
    a["label_ai"] = a.event_id.map(j.label); a["note_ai"] = a.event_id.map(j.note)
    a.to_csv(A / "label_A_merge_random.csv", index=False)
b = pd.read_csv(A / "label_B_duplicates.csv")
if (A / "ai_labels/B.json").exists():
    j = pd.DataFrame(json.load(open(A / "ai_labels/B.json"))).set_index(["event_1", "event_2"])
    key = list(zip(b.event_1, b.event_2))
    b["label_ai"] = [j.label.get(k) for k in key]; b["note_ai"] = [j.note.get(k) for k in key]
    b.to_csv(A / "label_B_duplicates.csv", index=False)
c = pd.read_csv(A / "label_C_imputed.csv")
parts = [pd.DataFrame(json.load(open(f))) for f in sorted(glob.glob(str(A / "ai_labels/C_*.json")))]
if parts:
    j = pd.concat(parts).drop_duplicates("row", keep="last").set_index("row")
    c["label_ai"] = c.row.map(j.label); c["note_ai"] = c.row.map(j.note)
    c["evidence_ai"] = c.row.map(j.get("evidence_url", pd.Series(dtype=str)))
    c.to_csv(A / "label_C_imputed.csv", index=False)
for n, d in (("A", a), ("B", b), ("C", c)):
    if "label_ai" in d:
        print(n, d.label_ai.value_counts(dropna=False).to_dict())
