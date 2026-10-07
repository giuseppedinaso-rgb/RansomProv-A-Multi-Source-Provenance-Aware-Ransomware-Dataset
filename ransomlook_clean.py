import html
"""Pulizia dei post di RansomLook e riallineamento alle altre due fonti.

Il `post_title` di RansomLook e' il testo grezzo del leak site, non un nome
di vittima: contiene suffissi di aggiornamento ("(UPDATE)", "PART 2"), volumi
("8TB internal documents"), percorsi, IP, bandiere-emoji, codici paese.
Qui lo si riduce a (nome, dominio, indizio di paese) e si allinea il nome
della gang a quello usato da ransomfeed / ransomware.live.
"""
import re, difflib, unicodedata

# ---------------------------------------------------------------- vittima
_NOISE = [
    r"\(\s*update[d]?\s*\)", r"\bupdate[d]?\b\s*[:\-]?",
    r"\bpart\s*\d+\b.*$", r"\bfiles?\s+part\s*\d+.*$",
    r"\bfull\s+leak\b", r"\bdata\s+leak\b", r"\bleak(ed)?\b\s*$",
    r"\b\d+(\.\d+)?\s*(tb|gb|mb)\b.*$",
    r"\b\d{1,3}(\.\d{1,3}){3}\b.*$",           # IP e tutto quel che segue
    r"[a-z]:\\\\.*$", r"[a-z]\$\\.*$",         # percorsi Windows
    r"\bnew\b\s*$", r"\bsold\b\s*$", r"\bfor sale\b.*$",
]
_DOMAIN = re.compile(r"\b((?:[a-z0-9-]+\.)+[a-z]{2,24})\b", re.I)
_CC_SUFFIX = re.compile(r",\s*([A-Z]{2})\s*$")


def _flag_to_cc(s):
    """🇨🇾 -> 'CY'. Coppie di regional indicator symbols."""
    ri = [c for c in s if 0x1F1E6 <= ord(c) <= 0x1F1FF]
    if len(ri) >= 2:
        return "".join(chr(ord(c) - 0x1F1E6 + ord("A")) for c in ri[:2])
    return None


def clean_title(title):
    """Restituisce dict(name, domain, cc, continuation)."""
    raw = unicodedata.normalize("NFKC", str(title or "")).replace("\xa0", " ").strip()
    cc = _flag_to_cc(raw)
    m = _CC_SUFFIX.search(raw)
    if m and not cc:
        cc = m.group(1)
        raw = raw[: m.start()]
    raw = "".join(c for c in raw if not (0x1F000 <= ord(c) <= 0x1FFFF))   # via emoji

    continuation = bool(re.fullmatch(r"\s*(part\s*\d+|update[d]?)\s*", raw, re.I))

    dom = None
    md = _DOMAIN.search(raw)
    if md:
        dom = md.group(1).lower().removeprefix("www.")
        # titolo troncato dal leak site subito dopo il dominio ("acme.ne...",
        # "novohamburgo.rs..."): l'ultima etichetta del dominio non e' affidabile
        if re.match(r"\s*(\.\.\.|…)", raw[md.end():]):
            dom = None

    s = raw.split("|")[0]                       # "x.com | 8TB documenti" -> "x.com"
    s = re.sub(r"\(([^)]*\.[a-z]{2,})\)", " ", s, flags=re.I)   # "(starbucks.com)"
    for p in _NOISE:
        s = re.sub(p, " ", s, flags=re.I)
    if dom:                                     # "Acme Inc acme.com" -> "Acme Inc"
        s = re.sub(re.escape(dom), " ", s, flags=re.I)
        s = re.sub(r"\bwww\.", " ", s, flags=re.I)
    s = re.sub(r"\bfiles?\b\s*$", " ", s, flags=re.I)
    s = re.sub(r"[\s\-–—:;,]+$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    name = s or dom or ""
    return {"name": name, "domain": dom, "cc": cc, "continuation": continuation}


_LEAK_TAIL = re.compile(
    r"\s*[-|:(]*\s*(?:\d+(?:[.,]\d+)?\s*(?:gb|tb|mb)\b.*|(?:has been |have been )?(?:hacked|leaked|"
    r"data (?:publication|leak(?:ed)?|stolen)|publication|full leak|leak)\b.*)$", re.I)


def clean_victim(v):
    """Nome leggibile della vittima. Il nome originale resta in victim_raw.
    Entita' HTML (anche annidate: &amp;amp;), URL -> host, etichette tra parentesi
    quadre all'inizio ([DISCLOSED], [EVIDENCE]), code tipo '1 TB data leaked'."""
    s = str(v or "")
    for _ in range(4):
        u = html.unescape(s)
        if u == s:
            break
        s = u
    m = re.match(r"^\s*[a-z][a-z0-9+.-]*://(?:www\.)?([^/\s]+)/?\s*$", s, re.I)
    if m:
        s = m.group(1)
    s = re.sub(r"^\s*(?:\[[^\]]*\]\s*)+", "", s)
    t = _LEAK_TAIL.sub("", s).strip(" -|:")
    s = t if len(t) >= 3 else s
    return re.sub(r"\s+", " ", s).strip() or str(v)


def victim_key(s):
    """Chiave di confronto per una vittima: minuscolo, senza www., TLD e
    punteggiatura, senza forme societarie comuni."""
    # stessa pulizia del nome mostrato nel rilascio: entita' HTML, etichette
    # [..], code tipo "HACKED. 50 GB DATA LEAKED". Senza, lo stesso attacco
    # pubblicato due volte con titoli diversi non veniva riconosciuto.
    s = clean_victim(s) if s else s
    s = unicodedata.normalize("NFKC", str(s or "")).lower().strip()
    # nome della vittima scritto come URL ("https://acme.com/"): si tiene
    # solo l'host, altrimenti la chiave diventa "https" per tutte
    m = re.match(r"^[a-z][a-z0-9+.-]*://([^/\s]+)", s)
    if m:
        s = m.group(1)
    s = s.removeprefix("www.")
    s = re.sub(r"\.(com|net|org|co|io|biz|info|[a-z]{2})(\.[a-z]{2})?$", "", s)
    s = re.sub(r"\b(inc|llc|ltd|limited|corp|corporation|co|gmbh|srl|spa|s\.?a|sas|bv|ag|plc|pty)\b\.?", " ", s)
    k = re.sub(r"[^a-z0-9]", "", s)
    # chiavi solo numeriche ("$150.000" -> "150000") non identificano una vittima
    return None if k in GENERIC_KEYS or k.isdigit() else k


# chiavi che non identificano una vittima: frammenti di URL, segnaposto e
# suffissi pubblici residui (gov.br -> "gov")
GENERIC_KEYS = {"https", "http", "www", "example", "test", "gov", "gob", "govt", "edu",
                "com", "org", "net", "k12", "mil", "ac", "nhs", "unknown", "none", "null"}


# ---------------------------------------------------------------- gang
def gang_key(s):
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


_GANG_SUFFIX = re.compile(r"(torrents?|data|blog|leaks?|news|team|group|\d+)$")


def build_gang_map(lookup_names, reference_names, cutoff=0.85):
    """Mappa ogni nome gang di RansomLook sul nome canonico delle altre fonti.
    Ordine: uguaglianza della chiave -> rimozione suffissi -> similarita'."""
    ref = {gang_key(n): n for n in reference_names if n}
    ref_keys = list(ref)
    out, how = {}, {}
    for n in set(lookup_names):
        k = gang_key(n)
        if k in ref:
            out[n], how[n] = ref[k], "exact"; continue
        k2 = k
        while _GANG_SUFFIX.search(k2) and len(k2) > 3:
            k2 = _GANG_SUFFIX.sub("", k2)
            if k2 in ref:
                break
        if k2 in ref:
            out[n], how[n] = ref[k2], "suffix"; continue
        best = difflib.get_close_matches(k, ref_keys, n=1, cutoff=cutoff)
        if best:
            out[n], how[n] = ref[best[0]], "fuzzy"; continue
        out[n], how[n] = n, "unmatched"
    return out, how
