#!/usr/bin/env python3
"""Precisione di ogni regola di inferenza dello stadio 4 -> build/report_rules.json.

Metodo: su ogni evento con valore OSSERVATO si finge che il valore manchi e si
applica la regola, confrontando la risposta con il valore osservato.

  inherited_victim  leave-one-out: il voto di maggioranza usa solo gli ALTRI
                    eventi osservati che condividono una chiave vittima o
                    dominio (l'evento stesso e' escluso dal conteggio);
  title_hint        paese estratto dal titolo del post (bandiera, ", CC");
  cctld             ccTLD del dominio;
  description       un solo paese menzionato nella descrizione del leak.

Ogni regola e' valutata da sola (dove scatta) e poi in cascata, nell'ordine in
cui lo stadio 4 la applica. Per ciascuna si riportano anche:
  * intervallo di confidenza di Wilson al 95%;
  * baseline di maggioranza sullo stesso sottoinsieme (rispondere sempre con
    la classe piu' frequente), per mostrare che la precisione non viene dalla
    sola prevalenza di US / Professional Services.

Limite: gli eventi con valore osservato non sono un campione casuale di
quelli mancanti (es. piu' recenti, fonti piu' ricche): la precisione misurata
qui e' una stima, non una garanzia sugli eventi effettivamente imputati.
"""
import json, math
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
from normalize import cctld_to_iso2, single_country_mention

BUILD = Path("build")


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round((c - h) * 100, 1), round((c + h) * 100, 1)]


def score(pred, true):
    m = pred.notna()
    n, k = int(m.sum()), int((pred[m] == true[m]).sum())
    base = round(true[m].value_counts(normalize=True).iloc[0] * 100, 1) if n else None
    return {"n": n, "coverage_pct": round(n / len(pred) * 100, 1),
            "precision_pct": round(k / n * 100, 1) if n else None, "ci95": wilson(k, n),
            "majority_baseline_pct": base}


def loo_inherit(ev, attr):
    obs = ev[ev[f"{attr}_src"] == "observed"]
    table = defaultdict(Counter)
    for keys, v in zip(obs["keys"], obs[attr]):
        for k in keys:
            table[k][v] += 1
    out = {}
    for e, keys, v in zip(obs.event_id, obs["keys"], obs[attr]):
        c = Counter()
        for k in keys:
            c.update(table.get(k, {}))
            c[v] -= 1                      # toglie il contributo dell'evento stesso
        c = +c                             # elimina conteggi <= 0
        out[e] = c.most_common(1)[0][0] if c else None
    return pd.Series(out)


def main():
    ev = pd.read_json(BUILD / "events.json")
    keys = pd.read_json(BUILD / "event_keys.json").set_index("event_id")
    ev = ev.join(keys, on="event_id")
    # chiavi uniche per evento: se vkey e dkey coincidono ("xgenize") l'evento
    # non deve contare due volte, altrimenti il leave-one-out lascia il suo voto
    ev["keys"] = [sorted(set(a) | set(b)) for a, b in zip(ev._vkeys, ev._dkeys)]

    rep = {"country": {}, "sector": {}}
    obs = ev[ev.country_src == "observed"].set_index("event_id")
    true = obs.country
    rules = {
        "inherited_victim": loo_inherit(ev, "country").reindex(obs.index),
        "title_hint": obs._hint.where(obs._hint.astype(bool), None),
        "cctld": obs.domain.map(cctld_to_iso2),
        "description": obs._desc.map(single_country_mention),
    }
    for name, pred in rules.items():
        rep["country"][name] = score(pred, true)
    casc, which = pd.Series(None, index=obs.index, dtype=object), pd.Series(None, index=obs.index, dtype=object)
    for name, pred in rules.items():
        fill = casc.isna() & pred.notna()
        casc[fill], which[fill] = pred[fill], name
    rep["country"]["cascade"] = score(casc, true)
    rep["country"]["cascade_by_rule"] = {n: score(casc[which == n], true[which == n]) for n in rules}

    obs_s = ev[ev.sector_src == "observed"].set_index("event_id")
    rep["sector"]["inherited_victim"] = score(loo_inherit(ev, "sector").reindex(obs_s.index), obs_s.sector)
    # stessa regola, solo dove le fonti dell'evento concordano (etichetta affidabile)
    agree = ~obs_s.sector_conflict
    rep["sector"]["inherited_victim_agree_only"] = score(
        loo_inherit(ev, "sector").reindex(obs_s.index)[agree], obs_s.sector[agree])

    json.dump(rep, open(BUILD / "report_rules.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    for attr in rep:
        print(f"\n{attr}")
        for name, r in rep[attr].items():
            if name == "cascade_by_rule":
                continue
            print(f"  {name:28} n={r['n']:6d} copertura {r['coverage_pct']:5.1f}%  "
                  f"precisione {r['precision_pct']}% {r['ci95']}  baseline {r['majority_baseline_pct']}%")
    print("-> build/report_rules.json")


if __name__ == "__main__":
    main()
