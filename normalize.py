"""Normalizzazione di paesi (ISO 3166-1 alpha-2) e nomi di gang, condivisa dagli stadi."""
import re, difflib, unicodedata
import pycountry

# Varianti che pycountry non risolve da solo, viste nei feed.
_COUNTRY_ALIASES = {
    "usa": "US", "united states of america": "US", "u.s.": "US", "u.s.a.": "US",
    "uk": "GB", "england": "GB", "scotland": "GB", "wales": "GB", "great britain": "GB",
    "south korea": "KR", "korea": "KR", "north korea": "KP", "russia": "RU",
    "iran": "IR", "vietnam": "VN", "taiwan": "TW", "syria": "SY", "laos": "LA",
    "bolivia": "BO", "venezuela": "VE", "tanzania": "TZ", "moldova": "MD",
    "czech republic": "CZ", "czechia": "CZ", "macedonia": "MK", "north macedonia": "MK",
    "ivory coast": "CI", "cote d'ivoire": "CI", "turkey": "TR", "türkiye": "TR",
    "uae": "AE", "the netherlands": "NL", "holland": "NL", "hong kong": "HK",
    "macau": "MO", "palestine": "PS", "kosovo": "XK", "brasil": "BR",
    "deutschland": "DE", "ukraina": "UA", "lettonia": "LV", "giordania": "JO",
    "bosnia": "BA", "the bahamas": "BS", "bahamas": "BS", "cape verde": "CV",
    "swaziland": "SZ", "burma": "MM", "vatican": "VA", "micronesia": "FM",
    "saint kitts and nevis": "KN", "antigua and barbuda": "AG",
    "democratic republic of the congo": "CD", "dr congo": "CD", "republic of the congo": "CG",
    # nomi in italiano presenti in ransomfeed
    "marocco": "MA", "portorico": "PR", "principaut de monaco": "MC",
    "papua nuova guinea": "PG", "repubblica di palau": "PW", "principato di andorra": "AD",
    "republica de cabo verde": "CV", "cipro": "CY",
    "repubblica turca di cipro del nord": "CY",   # non ISO: ricondotto a CY
}
# Segnaposto che significano "sconosciuto".
_COUNTRY_NULL = {"", "non disponibile", "n/a", "unknown", "none", "null", "-", "??", "xx", "zz"}

_cache = {}


def to_iso2(value):
    """Stringa di paese qualsiasi -> alpha-2, oppure None se sconosciuto o non risolvibile."""
    if value is None:
        return None
    raw = unicodedata.normalize("NFKC", str(value)).strip()
    low = raw.lower()
    if low in _COUNTRY_NULL:
        return None
    if low in _cache:
        return _cache[low]
    iso = None
    if low in _COUNTRY_ALIASES:
        iso = _COUNTRY_ALIASES[low]
    elif len(raw) == 2 and raw.isalpha():
        iso = raw.upper() if pycountry.countries.get(alpha_2=raw.upper()) or raw.upper() == "XK" else None
    elif len(raw) == 3 and raw.isalpha() and pycountry.countries.get(alpha_3=raw.upper()):
        iso = pycountry.countries.get(alpha_3=raw.upper()).alpha_2
    else:
        try:
            iso = pycountry.countries.lookup(raw).alpha_2
        except LookupError:
            try:
                iso = pycountry.countries.search_fuzzy(raw)[0].alpha_2
            except LookupError:
                iso = None
    _cache[low] = iso
    return iso


# ccTLD che NON indicano un paese (uso commerciale generico)
_GENERIC_CCTLD = {"co", "io", "ai", "me", "tv", "cc", "ws", "to", "ly", "fm", "gg",
                  "sh", "so", "la", "eu", "app", "dev",
                  # v1.1: ccTLD usati in modo generico o da domini gratuiti (audit del 4/10/2026)
                  "ms", "nu", "tk", "ga", "cf", "gq", "pw", "vc", "st"}
# v1.1: governi locali USA con sigla di Stato come ultima etichetta (ci.hercules.ca = California)
_US_LOCAL_GOV = re.compile(r"^(ci|co|city|cityof|town|townof|twp|vil|village|county)\.[a-z0-9-]+\.(al|ak|az|ar|ca|co|ct|de|fl|ga|hi|id|il|in|ia|ks|ky|la|me|md|ma|mi|mn|ms|mo|mt|ne|nv|nh|nj|nm|ny|nc|nd|oh|ok|or|pa|ri|sc|sd|tn|tx|ut|vt|va|wa|wv|wi|wy)(\.us)?$")


def cctld_to_iso2(domain):
    if not domain:
        return None
    if _US_LOCAL_GOV.match(str(domain).lower()):
        return "US"
    m = re.search(r"\.([a-z]{2})$", str(domain).lower())
    if not m or m.group(1) in _GENERIC_CCTLD:
        return None
    t = m.group(1).upper()
    t = "GB" if t == "UK" else t
    return t if pycountry.countries.get(alpha_2=t) else None


# ---------------------------------------------------------------- gang
def gang_key(s):
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


_SERVICE = re.compile(r"(torrents?|data|blog|leaks?|news|team|group|ransomware)$")
_DIGITS = re.compile(r"\d+$")


def canonical_gang_map(names_by_source, cutoff=0.85):
    """Allinea le varianti di scrittura di una gang fra le fonti.

    Regole, in ordine, volutamente conservative:
      1. stessa chiave alfanumerica (maiuscole, spazi, punteggiatura);
      2. suffisso di servizio del leak site ('torrents', 'data', 'blog',
         'team', 'group'...): 'clop torrents' -> 'clop';
      3. cifre finali SOLO se la forma con cifre non compare mai in
         ransomfeed/ransomware.live e quella senza si': e' allora un artefatto
         di nomenclatura di RansomLook ('killsec3' -> 'killsec'). Se entrambe
         compaiono nelle fonti ricche restano distinte: 'lockbit3' e
         'lockbit5' sono versioni diverse del brand.
    Nessuna fusione per somiglianza: le coppie simili vengono solo proposte
    per revisione manuale (blackwater/blackmatter sono gruppi diversi).
    """
    from collections import Counter
    rich = {gang_key(n) for src in ("ransomfeed", "ransomlive")
            for n in names_by_source.get(src, []) if n}
    allk = Counter(gang_key(n) for src in names_by_source
                   for n in names_by_source[src] if n)
    keys = [k for k in allk if k]
    canon, rule = {}, {}
    for k in keys:
        canon[k], rule[k] = k, "identity"
    for k in keys:
        b = _SERVICE.sub("", k)
        if b != k and b in allk and len(b) >= 3:
            canon[k], rule[k] = b, "service_suffix"; continue
        d = _DIGITS.sub("", k)
        if d != k and len(d) >= 3 and d in rich and k not in rich:
            canon[k], rule[k] = d, "lookup_digit_suffix"
    # proposte di revisione, NON applicate
    targets = sorted(set(canon.values()))
    review = []
    for k in targets:
        if len(k) < 5:
            continue
        for m in difflib.get_close_matches(k, [t for t in targets if t != k and len(t) >= 5],
                                           n=3, cutoff=cutoff):
            if k < m:
                review.append((k, m, round(difflib.SequenceMatcher(None, k, m).ratio(), 3)))
    return canon, rule, review


# ---------------------------------------------------------------- paese citato nel testo
_NAMES = None


def _country_names():
    global _NAMES
    if _NAMES is None:
        _NAMES = {}
        for c in pycountry.countries:
            n = [c.name] + ([c.common_name] if hasattr(c, "common_name") else [])
            _NAMES[c.alpha_2] = n
        _NAMES["US"] += ["United States", "USA", "U.S."]
        _NAMES["GB"] += ["United Kingdom", "UK", "England", "Scotland", "Wales"]
        _NAMES["KR"] += ["South Korea"]; _NAMES["RU"] += ["Russia"]
        _NAMES["VN"] += ["Vietnam"];     _NAMES["TW"] += ["Taiwan"]
        _NAMES["IR"] += ["Iran"];        _NAMES["CZ"] += ["Czech Republic"]
    return _NAMES


# Nomi di paese che coincidono con stati USA o parti di toponimi: si accettano
# solo con un contesto che li disambigua.
_AMBIGUOUS = {
    "GE": r"Tbilisi|Republic of Georgia|Georgian (?:company|government)",
    "JE": r"(?<!New )Jersey,? Channel Islands|St\.? Helier",
    "GN": r"Conakry|Republic of Guinea",
    "PG": r"Papua New Guinea|Port Moresby",
}


def mentions_country(text, iso2):
    """True se `text` nomina il paese `iso2` come parola intera."""
    if not text or not iso2:
        return False
    if iso2 in _AMBIGUOUS:
        return re.search(_AMBIGUOUS[iso2], text) is not None
    return any(re.search(r"(?<![A-Za-z])" + re.escape(n) + r"(?![A-Za-z])", text)
               for n in _country_names().get(iso2, []))


def single_country_mention(text, min_len=20):
    """Il paese nominato nel testo, se e' uno e uno solo; altrimenti None."""
    if not text or len(text) < min_len:
        return None
    hits = [c for c in _country_names() if mentions_country(text, c)]
    return hits[0] if len(hits) == 1 else None
