"""Protocollo comune "prossimo evento" per i modelli multi-attributo.

Ogni gang e' una sequenza di eventi in ordine di data (poi event_id). Ogni
evento ha tre attributi categorici da prevedere:
  delta    intervallo dall'evento precedente della stessa gang:
           0 = primo evento, 1 = stesso giorno, 2 = entro 7 giorni, 3 = oltre
  country  i 30 paesi piu' frequenti nel training, piu' 'other' e 'unknown'
  sector   13 classi piu' 'unknown'
Compito: dato tutto il passato della gang, prevedere gli attributi dell'evento
successivo.

Divisione per data dell'evento DA PREVEDERE: training < 2025-01-01,
validazione 2025-01-01 .. 2025-06-30, test >= 2025-07-01. Il contesto e' sempre
tutta la storia precedente, a prescindere dal periodo.

Etichette: per training e validazione quelle della versione scelta (observed o
enriched); per il TEST sempre quelle osservate; 'unknown' non e' mai un
bersaglio (ignorato nella loss e nelle metriche).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, top_k_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
VAL_START, TEST_START = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-07-01")
FIELDS = ["delta", "country", "sector"]
IGNORE = -100


def delta_bucket(days):
    return np.select([np.isnan(days), days == 0, days <= 7], [0, 1, 2], 3)


def source_attributes(source):
    """Paese e settore come li riporta UNA sola fonte (moda dei suoi record per evento)."""
    r = pd.read_json(ROOT / "build" / "record_to_event.json", dtype={"source_id": str})
    u = pd.read_json(ROOT / "build" / "unified.json", dtype={"source_id": str})
    u = u[u.source == source].merge(r.drop_duplicates(["source", "source_id"]), on=["source", "source_id"])
    mode = lambda x: x.dropna().mode().iloc[0] if x.notna().any() else None
    return u.groupby("event_id").agg(country_src1=("country", mode), sector_src1=("sector", mode))


def load(version, source=None):
    """source: se indicato, vista a fonte singola. Si tengono solo gli eventi visti
    da quella fonte, e paese/settore di training/validazione sono quelli che
    riporta quella fonte. Il test resta valutato sulle etichette osservate del
    dataset unito (stessa verita' per tutte le viste)."""
    ev = pd.read_json(ROOT / "release" / f"events_{version}.json")
    if source:
        ev = ev[ev.sources.map(lambda s: source in s)].copy()
        sa = source_attributes(source)
        ev["country"] = ev.event_id.map(sa.country_src1); ev["sector"] = ev.event_id.map(sa.sector_src1)
    obs = pd.read_json(ROOT / "release" / "events_observed.json")[["event_id", "country", "sector"]]
    ev = ev.merge(obs.rename(columns={"country": "country_obs", "sector": "sector_obs"}), on="event_id")
    ev["d"] = pd.to_datetime(ev.discovered)
    ev = ev.sort_values(["gang", "d", "event_id"]).reset_index(drop=True)
    ev.attrs["source"] = source or "merged"
    ev["delta_days"] = ev.groupby("gang").d.diff().dt.days
    ev["delta"] = delta_bucket(ev.delta_days.values.astype(float))
    ev["gap"] = np.log1p(ev.delta_days.fillna(0).clip(lower=0))
    ev["split"] = np.where(ev.d < VAL_START, "train", np.where(ev.d < TEST_START, "val", "test"))
    train = ev[ev.split == "train"]
    top_c = list(obs.merge(ev[["event_id", "split"]], on="event_id").query("split == 'train'")
                 .country.value_counts().head(30).index)
    vocab = {"delta": ["first", "same_day", "le_7d", "gt_7d"],
             "country": top_c + ["other", "unknown"],
             "sector": sorted(obs.sector.dropna().unique()) + ["unknown"]}
    ci = {c: i for i, c in enumerate(vocab["country"])}
    si = {s: i for i, s in enumerate(vocab["sector"])}
    cmap = lambda c: ci["unknown"] if pd.isna(c) else ci.get(c, ci["other"])
    smap = lambda s: si["unknown"] if pd.isna(s) else si[s]
    ev["country_i"], ev["sector_i"] = ev.country.map(cmap), ev.sector.map(smap)
    ev["country_obs_i"], ev["sector_obs_i"] = ev.country_obs.map(cmap), ev.sector_obs.map(smap)
    ev["gang_i"] = ev.gang.astype("category").cat.codes + 1
    return ev, vocab


def targets(ev, vocab):
    """Bersagli per ogni evento: version labels per train/val, osservate per test."""
    unk = {"country": vocab["country"].index("unknown"), "sector": vocab["sector"].index("unknown")}
    out = {"delta": ev.delta.values.copy()}
    for f in ("country", "sector"):
        lab = np.where(ev.split == "test", ev[f"{f}_obs_i"], ev[f"{f}_i"]).astype(int)
        lab[lab == unk[f]] = IGNORE
        out[f] = lab
    out["delta"][ev.delta.values == 0] = IGNORE          # il primo evento non ha intervallo
    return out


def metrics(y_true, y_pred, proba=None):
    """proba (opzionale): (n, K) probabilita' su tutte le classi del vocabolario;
    aggiunge log-loss e accuratezza top-3 (metriche probabilistiche)."""
    m = y_true != IGNORE
    yt, yp = y_true[m], y_pred[m]
    if not m.any():                                  # nessun bersaglio (es. vista solo RansomLook)
        return {"n": 0, "accuracy": float("nan"), "macro_f1": float("nan"), "weighted_f1": float("nan"),
                "log_loss": float("nan"), "top3": float("nan")}
    out = {"n": int(m.sum()), "accuracy": accuracy_score(yt, yp),
           "macro_f1": f1_score(yt, yp, average="macro"), "weighted_f1": f1_score(yt, yp, average="weighted")}
    if proba is not None:
        P = np.clip(proba[m], 1e-9, 1); P = P / P.sum(1, keepdims=True); K = P.shape[1]
        out["log_loss"] = log_loss(yt, P, labels=np.arange(K))
        out["top3"] = top_k_accuracy_score(yt, P, k=3, labels=np.arange(K))
    return out
