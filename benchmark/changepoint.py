#!/usr/bin/env python3
"""Benchmark 7 — rilevamento di cambi di regime e operazioni di polizia.

Domanda: le serie del dataset "vedono" eventi reali noti? Per ogni gang colpita
da un'operazione di polizia o da una chiusura documentata si prende la serie
settimanale delle sue rivendicazioni (52 settimane prima e dopo l'evento) e si
applicano algoritmi di change-point senza conoscere la data.

Eventi di riferimento (data dell'intervento sul leak site):
  verificati su fonti giornalistiche (ottobre 2026):
    hive          2023-01-26  takedown FBI
    ragnarlocker  2023-10-19  sequestro Europol (16-20 ottobre)
    lockbit3      2024-02-19  Operation Cronos (NCA)
    8base         2025-02-10  operazione Phobos/8Base (arresti in Thailandia)
    blacksuit     2025-07-24  Operation Checkmate
  da verificare (data nota ma non ricontrollata in questa sessione):
    alphv         2023-12-19  disruption FBI (poi exit scam marzo 2024)
    dispossessor  2024-08-12  takedown FBI
    ransomhub     2025-04-01  leak site offline
    conti         2022-05-19  chiusura annunciata

Metodi (ruptures): PELT (costo rbf e l2), Binary Segmentation (l2), Window
(l2); penalita' = 2*log(n)*var. Baseline "Drop": settimana dopo cui la media
mobile di 4 settimane scende sotto il 30% della media delle 12 precedenti.
Metriche: rilevamento entro +-4 settimane (hit), ritardo in settimane,
e falsi allarmi per anno (change-point lontani dall'evento).
Uscita: benchmark/results/changepoint.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import ruptures as rpt

ROOT = Path(__file__).resolve().parents[1]
EVENTS = [("hive", "2023-01-26", True), ("ragnarlocker", "2023-10-19", True), ("lockbit3", "2024-02-19", True),
          ("8base", "2025-02-10", True), ("blacksuit", "2025-07-24", True), ("alphv", "2023-12-19", False),
          ("dispossessor", "2024-08-12", False), ("ransomhub", "2025-04-01", False), ("conti", "2022-05-19", False)]
TOL = 4


def series(ev, gang, date):
    d0 = pd.Timestamp(date)
    w = pd.date_range(d0 - pd.Timedelta(weeks=52), d0 + pd.Timedelta(weeks=52), freq="W-MON")
    x = ev[ev.gang == gang]
    counts = x.groupby(x.d.dt.to_period("W-SUN").dt.start_time).size().reindex(w, fill_value=0)
    return w, counts.values.astype(float)


def detect(y, method):
    n = len(y); s = y.reshape(-1, 1)
    pen = 2 * np.log(n) * max(np.var(y), 1e-6)
    if method == "Drop":
        m4 = pd.Series(y).rolling(4).mean(); m12 = pd.Series(y).shift(4).rolling(12).mean()
        idx = np.where((m12 > 1) & (m4 < 0.3 * m12))[0]
        return [int(idx[0]) - 3] if len(idx) else []
    algo = {"PELT-rbf": rpt.Pelt(model="rbf"), "PELT-l2": rpt.Pelt(model="l2"),
            "BinSeg-l2": rpt.Binseg(model="l2"), "Window-l2": rpt.Window(width=8, model="l2")}[method]
    pen_m = pen if method != "PELT-rbf" else 2 * np.log(n)
    return [b for b in algo.fit(s).predict(pen=pen_m) if b < n]


def main():
    ev = pd.read_json(ROOT / "release" / "events_enriched.json")
    ev["d"] = pd.to_datetime(ev.discovered)
    rows = []
    for gang, date, verified in EVENTS:
        w, y = series(ev, gang, date)
        if y.sum() < 10:
            print("poche rivendicazioni:", gang, y.sum()); continue
        k = int(np.argmin(np.abs(w - pd.Timestamp(date))))
        for method in ("PELT-rbf", "PELT-l2", "BinSeg-l2", "Window-l2", "Drop"):
            cps = detect(y, method)
            near = [c for c in cps if abs(c - k) <= TOL]
            lag = min(near, key=lambda c: abs(c - k)) - k if near else None
            rows.append({"gang": gang, "event_date": date, "verified": verified, "method": method,
                         "claims_before": int(y[:k].sum()), "claims_after": int(y[k:].sum()),
                         "hit": bool(near), "lag_weeks": lag, "n_cps": len(cps),
                         "false_alarms_per_year": (len(cps) - (1 if near else 0)) / (len(y) / 52)})
    R = pd.DataFrame(rows)
    R.to_csv(ROOT / "benchmark" / "results" / "changepoint.csv", index=False)
    print(R.pivot_table(index="gang", columns="method", values="hit", aggfunc="first").to_string())
    print(R.groupby("method").agg(hit_rate=("hit", "mean"), mean_abs_lag=("lag_weeks", lambda s: s.dropna().abs().mean()),
                                  false_alarms_per_year=("false_alarms_per_year", "mean")).round(2).to_string())
    print(R.drop_duplicates("gang")[["gang", "event_date", "verified", "claims_before", "claims_after"]].to_string(index=False))


if __name__ == "__main__":
    main()
