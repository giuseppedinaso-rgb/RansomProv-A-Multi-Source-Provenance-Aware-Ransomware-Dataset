#!/usr/bin/env python3
"""Stadio 5a — paese dal sito web della vittima, in modo deterministico.

Per ogni evento con dominio:
  1. snapshot su archive.org PRECEDENTE alla data dell'attacco (CDX API), cosi'
     la pagina non e' quella eventualmente defacciata o offline dopo l'attacco
     e il risultato e' riproducibile: lo snapshot e' fisso. Se non esiste, il
     sito dal vivo come ripiego, marcato;
  2. home page + al piu' una pagina contatti/impressum/about dello stesso snapshot;
  3. segnali di paese, ciascuno registrato separatamente:
       schema      indirizzo strutturato schema.org (addressCountry)
       vat         identificativi fiscali/societari tipici di un paese
       address     indirizzo postale in formato inconfondibile (USA: 'TX 78701',
                   Canada: 'ON M5V 3L9', Regno Unito: postcode completo)
       phone       numeri internazionali (+CC), link tel:, formato NANP '(555) 123-4567';
                   esclusi i numeri non geografici (numeri verdi, premium, VoIP)
  4. decisione: si assegna un paese solo se i segnali FORTI (schema, vat, address)
     concordano, oppure, in loro assenza, se i numeri di telefono indicano
     tutti lo stesso paese. Altrimenti nessuna risposta.

Salva anche il testo visibile (titolo, meta description, intestazioni,
primi caratteri) in build/site_text.jsonl, per classificare il settore in un
passo successivo con un modello senza ricerca web.

Modalita':
  validate  eventi con paese osservato e unanime: misura la precisione di
            ogni segnale e della regola di decisione;
  apply     eventi con paese o settore mancante.
"""
import re, json, time, argparse, random, threading, socket
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import requests, pandas as pd, phonenumbers
from bs4 import BeautifulSoup
from normalize import to_iso2

BUILD = Path("build")
UA = {"User-Agent": "Mozilla/5.0 (research; ransomware-dataset-enrichment)"}
CDX = "https://web.archive.org/cdx/search/cdx"
AVAIL = "https://archive.org/wayback/available"
CONTACT = re.compile(r"contact|kontakt|contatti|contacto|impressum|imprint|about|chi-siamo|"
                     r"qui-sommes|uber-uns|legal|mentions", re.I)

# identificativi che legano un'azienda a un paese: (paese, regex)
VAT = [
    ("IT", r"\bP\.?\s?IVA\b|\bPartita\s+IVA\b|\bC\.?F\.?\s*/\s*P\.?\s?IVA\b"),
    ("DE", r"\bUSt-?Id(?:Nr)?\.?\b|\bHandelsregister\b|\bAmtsgericht\b|\bHRB\s?\d"),
    ("AT", r"\bFirmenbuch(?:nummer)?\b|\bFN\s?\d{5,6}\s?[a-z]\b"),
    ("FR", r"\bSIRET\b|\bSIREN\b|\bRCS\s+[A-Z][a-z]+"),
    ("ES", r"\bC\.?I\.?F\.?\s*:?\s*[A-Z]\d{7}[A-Z0-9]\b"),
    ("BR", r"\bCNPJ\b"),
    ("AU", r"\bA\.?B\.?N\.?\s*:?\s*\d{2}\s?\d{3}\s?\d{3}\s?\d{3}\b"),
    ("GB", r"Registered in England|Company (?:Registration )?No\.?\s*\d{8}|Companies House"),
    ("NL", r"\bKvK(?:-nummer)?\b|\bKamer van Koophandel\b"),
    ("BE", r"\bBTW\s*:?\s*BE\s?0\d{3}|\bTVA\s*:?\s*BE\s?0\d{3}"),
    ("CH", r"\bCHE-\d{3}\.\d{3}\.\d{3}\b"),
    ("IN", r"\bGSTIN\b|\bCIN\s*:?\s*[LU]\d{5}"),
    ("CA", r"\bGST/HST\b|\bBusiness Number\b"),
    ("MX", r"\bR\.?F\.?C\.?\s*:?\s*[A-Z&]{3,4}\d{6}"),
]
US_STATES = ("AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV "
             "NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC").split()
ADDRESS = [
    # "Austin, TX 78701" / "Austin TX 78701-1234"
    ("US", re.compile(r"\b[A-Z][a-zA-Z .'-]{2,30},?\s+(?:" + "|".join(US_STATES) + r")\s+\d{5}(?:-\d{4})?\b")),
    # "Toronto, ON M5V 3L9"
    ("CA", re.compile(r"\b(?:AB|BC|MB|NB|NL|NS|NT|NU|ON|PE|QC|SK|YT)\s+[A-Z]\d[A-Z]\s?\d[A-Z]\d\b")),
    # "London SW1A 1AA" (postcode UK completo)
    ("GB", re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s\d[A-Z]{2}\b(?=.{0,40}(?:United Kingdom|UK|England|Scotland|Wales|London))")),
]
_tls = threading.local()


def sess():
    if not hasattr(_tls, "s"):
        _tls.s = requests.Session(); _tls.s.headers.update(UA)
    return _tls.s


def get(url, timeout=25, tries=3):
    for k in range(tries):
        try:
            r = sess().get(url, timeout=timeout, allow_redirects=True)
            if r.status_code == 200:
                # requests assume ISO-8859-1 se l'header non dichiara il charset:
                # in quel caso usiamo la codifica rilevata dal contenuto.
                if not r.encoding or r.encoding.lower() == "iso-8859-1":
                    r.encoding = r.apparent_encoding
                return r
            if r.status_code in (429, 503):
                time.sleep(10 * (k + 1)); continue
            return None
        except Exception:
            time.sleep(3 * (k + 1))
    return None


def snapshot(domain, before):
    """(url_da_scaricare, origine) dello snapshot piu' recente prima di `before`.

    Prima l'API availability (~1-2 s), che restituisce lo snapshot piu' vicino:
    lo accettiamo solo se strettamente precedente a `before`. Altrimenti CDX,
    che su domini con molte catture impiega 30-60 s: timeout ampio e finestra
    limitata agli ultimi 3 anni. L'origine distingue 'nessuno snapshot' da
    'archivio non raggiungibile', per poterli valutare separatamente.
    """
    r = get(f"{AVAIL}?url={domain}&timestamp={before}", timeout=30)
    if r is not None:
        try:
            c = r.json().get("archived_snapshots", {}).get("closest")
            if c and c.get("status") == "200" and c["timestamp"][:8] < before:
                ts = c["timestamp"]
                orig = c["url"].split(f"/{ts}/", 1)[1]
                return f"https://web.archive.org/web/{ts}id_/{orig}", f"archive:{ts}"
        except (ValueError, KeyError, IndexError):
            pass
    start = str(int(before[:4]) - 3)
    r = get(f"{CDX}?url={domain}&from={start}&to={before}&limit=-1&filter=statuscode:200"
            f"&filter=mimetype:text/html&output=json&fl=timestamp,original", timeout=120, tries=2)
    if r is None:
        return f"https://{domain}/", "live_cdx_fail"
    try:
        rows = r.json()
        if len(rows) > 1:
            ts, orig = rows[-1]
            if ts[:8] < before:
                return f"https://web.archive.org/web/{ts}id_/{orig}", f"archive:{ts}"
    except ValueError:
        pass
    return f"https://{domain}/", "live_nosnap"


def signals(html):
    soup = BeautifulSoup(html, "lxml")
    sig = {"schema": [], "vat": [], "address": [], "phone": []}
    # schema.org JSON-LD e microdata
    for tag in soup.find_all("script", type="application/ld+json"):
        for m in re.finditer(r'"addressCountry"\s*:\s*(?:\{[^}]*"name"\s*:\s*)?"([^"]+)"', tag.string or ""):
            iso = to_iso2(m.group(1))
            if iso: sig["schema"].append(iso)
    for el in soup.select('[itemprop="addressCountry"]'):
        iso = to_iso2(el.get("content") or el.get_text(" ", strip=True))
        if iso: sig["schema"].append(iso)
    text = soup.get_text(" ", strip=True)
    for iso, pat in VAT:
        if re.search(pat, text):
            sig["vat"].append(iso)
    for iso, rx in ADDRESS:
        if rx.search(text):
            sig["address"].append(iso)
    # telefono: solo formato internazionale esplicito, e link tel:
    cands = re.findall(r"\+\d[\d\s().-]{7,18}\d", text)
    cands += [a.get("href", "")[4:] for a in soup.select('a[href^="tel:"]') if a.get("href", "").startswith("tel:+")]
    nanp = re.findall(r"\(\d{3}\)\s?\d{3}[-.\s]\d{4}\b", text)     # (555) 123-4567
    NONGEO = {phonenumbers.PhoneNumberType.TOLL_FREE, phonenumbers.PhoneNumberType.PREMIUM_RATE,
              phonenumbers.PhoneNumberType.SHARED_COST, phonenumbers.PhoneNumberType.VOIP,
              phonenumbers.PhoneNumberType.UAN, phonenumbers.PhoneNumberType.PERSONAL_NUMBER}
    for c, region in [(c, None) for c in cands[:20]] + [(c, "US") for c in nanp[:10]]:
        try:
            n = phonenumbers.parse(c, region)
            if not phonenumbers.is_valid_number(n):
                continue
            if phonenumbers.number_type(n) in NONGEO:     # numeri verdi: non dicono il paese
                continue
            rc = phonenumbers.region_code_for_number(n)
            if rc: sig["phone"].append(rc)
        except phonenumbers.NumberParseException:
            pass
    # testo per il settore
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    md = soup.find("meta", attrs={"name": re.compile("^description$", re.I)})
    heads = " | ".join(h.get_text(" ", strip=True) for h in soup.find_all(["h1", "h2"])[:8])
    blurb = {"title": title[:200], "meta": (md.get("content", "") if md else "")[:400],
             "heads": heads[:600], "body": text[:1500]}
    links = [a.get("href") for a in soup.find_all("a", href=True) if CONTACT.search(a.get("href", ""))]
    return sig, blurb, links


def decide(sig):
    """Paese + regola usata, oppure (None, motivo)."""
    strong = sig["schema"] + sig["vat"] + sig.get("address", [])
    if strong:
        c = Counter(strong)
        if len(c) == 1:
            return next(iter(c)), "strong"
        return None, "strong_conflict"
    if sig["phone"]:
        c = Counter(sig["phone"])
        if len(c) == 1:
            return next(iter(c)), "phone"
        top, n = c.most_common(1)[0]
        if n / sum(c.values()) >= 0.8:
            return top, "phone_majority"
        return None, "phone_conflict"
    return None, "no_signal"


# Il campo domain contiene anche valori non utilizzabili (troncati "foo.co...",
# indirizzi .onion, frasi dei leak site, "u.s. gov"): li scartiamo senza rete.
DOMAIN_OK = re.compile(r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}")


def resolves(domain):
    """Il dominio ha ancora un record DNS? Evita tentativi lunghi su domini morti."""
    try:
        socket.getaddrinfo(domain, 443)
        return True
    except (socket.gaierror, UnicodeError, OSError):
        return False


def fetch_live(domain):
    if not resolves(domain):
        return None
    return get(f"https://{domain}/", timeout=15, tries=1)


def process(ev):
    out = {"event_id": ev["event_id"], "domain": ev["domain"]}
    if not DOMAIN_OK.fullmatch(ev["domain"] or "") or ev["domain"].endswith(".onion"):
        return {**out, "origin": "invalid_domain", "fetched": False}
    before = pd.Timestamp(ev["discovered"]).strftime("%Y%m%d")
    url, origin = snapshot(ev["domain"], before)
    if origin.startswith("archive"):
        r = get(url)
        if r is None:
            url, origin = f"https://{ev['domain']}/", "live_fallback"
            r = fetch_live(ev["domain"])
    else:
        r = fetch_live(ev["domain"])
    out["origin"] = origin
    if r is None:
        return {**out, "fetched": False}
    try:
        sig, blurb, links = signals(r.text)
    except Exception as e:
        return {**out, "fetched": True, "parse_error": str(e)[:120]}
    # al piu' una pagina contatti, dallo stesso snapshot
    if links and not (sig["schema"] or sig["vat"] or sig["address"]):
        href = links[0]
        if href.startswith("/"):
            base = url.split("id_/")[0] + "id_/" if "id_/" in url else ""
            host = url.split("id_/")[-1].split("/")[0:3]
            href = (base + "/".join(host) + href) if base else f"https://{ev['domain']}{href}"
        if href.startswith("http"):
            r2 = get(href)
            if r2 is not None:
                try:
                    s2, _, _ = signals(r2.text)
                    for k in sig: sig[k] += s2[k]
                except Exception:
                    pass
    country, rule = decide(sig)
    return {**out, "fetched": True, "signals": sig, "country": country, "rule": rule, "text": blurb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["validate", "apply"])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ev = pd.read_json(BUILD / "events.json")
    ev = ev[ev.domain.notna()]
    if a.mode == "validate":
        pool = ev[(ev.country_resolution == "unanimous")]
        sample = pool.sample(min(a.n, len(pool)), random_state=a.seed)
    else:
        sample = ev[(ev.country_src == "missing") | (ev.sector_src == "missing")]
    out = BUILD / f"site_{a.mode}.jsonl"
    done = {json.loads(l)["event_id"] for l in open(out)} if out.exists() else set()
    todo = [r for r in sample.to_dict("records") if r["event_id"] not in done]
    random.Random(a.seed).shuffle(todo)
    print(f"{a.mode}: campione {len(sample)} | fatti {len(done)} | da fare {len(todo)}", flush=True)

    t0 = time.time()
    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(process, r): r for r in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            r = futs[fu]
            try:
                res = fu.result()
            except Exception as e:
                res = {"event_id": r["event_id"], "domain": r["domain"], "error": str(e)[:150]}
            if a.mode == "validate":
                res["true_country"] = r["country"]
            res["ts"] = datetime.now().isoformat(timespec="seconds")
            f.write(json.dumps(res, ensure_ascii=False) + "\n"); f.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}  {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
