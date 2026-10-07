"""Abbinamento severo tra nomi di organizzazioni esterne ed eventi del dataset.

Chiave identica (nome o etichetta del dominio, normalizzati senza forme
societarie), oppure una contenuta nell'altra con almeno 6 caratteri e rapporto
di lunghezza >= 0.6, oppure similarita' >= 0.92; parole generiche escluse.
Finestra temporale configurabile intorno alla data di riferimento.
"""
import re
from difflib import SequenceMatcher
import pandas as pd

CORP = r"\b(inc|incorporated|corp|corporation|co|company|ltd|limited|llc|llp|pllc|plc|pc|pa|holdings?|group|the|sa|nv|ag|se|lp|dba)\b"
GENERIC = {"international", "financial", "services", "financialservices", "health", "healthcare", "medical",
           "group", "systems", "solutions", "technologies", "technology", "industries", "partners", "associates",
           "consulting", "management", "capital", "bank", "insurance", "school", "county", "city", "university"}


def norm(s):
    s = re.sub(r"\(.*?\)", " ", str(s or "").lower())
    s = re.sub(r"\.(com|net|org|io|co|us)\b", " ", s)
    s = re.sub(CORP, " ", s)
    return re.sub(r"[^a-z0-9]", "", s)


def prepare(ev):
    ev = ev.copy()
    ev["d"] = pd.to_datetime(ev.discovered)
    ev["k1"] = ev.victim.map(norm)
    ev["k2"] = ev.domain.fillna("").map(lambda d: norm(d.split(".")[0]) if d else "")
    return ev


def score(k, kk):
    if not kk or len(kk) < 3 or kk in GENERIC:
        return 0.0
    if kk == k:
        return 1.0
    short, long_ = sorted((k, kk), key=len)
    if len(short) >= 6 and short not in GENERIC and short in long_ and len(short) / len(long_) >= 0.6:
        return 0.95
    return SequenceMatcher(None, k, kk).ratio()


def match(name, date, ev, before=180, after=180, min_sim=0.92):
    k = norm(name)
    if len(k) < 3 or k in GENERIC:
        return None
    t = pd.Timestamp(date)
    w = ev[(ev.d >= t - pd.Timedelta(days=before)) & (ev.d <= t + pd.Timedelta(days=after))]
    # prefiltro economico: stessa iniziale
    w = w[(w.k1.str[:1] == k[:1]) | (w.k2.str[:1] == k[:1])]
    best = None
    for e in w.itertuples():
        s = max(score(k, e.k1), score(k, e.k2))
        if s >= min_sim and (best is None or s > best[0] or (s == best[0] and abs((e.d - t).days) < abs((best[1].d - t).days))):
            best = (s, e)
    return best
