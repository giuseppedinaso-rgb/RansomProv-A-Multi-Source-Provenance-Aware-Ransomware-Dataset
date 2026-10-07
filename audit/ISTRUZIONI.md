# Istruzioni per l'etichettatura

Tre file CSV. Ogni file ha due colonne da compilare: `label_A` (primo annotatore) e `label_B` (secondo annotatore). Gli annotatori lavorano **separatamente**, senza vedere le etichette dell'altro: l'accordo tra i due (kappa di Cohen) va riportato nel paper. Se c'è un solo annotatore, compilare solo `label_A`. La colonna `note` è libera.

Tempo stimato: A circa 30 minuti, B circa 30 minuti, C circa 2–3 ore (richiede ricerche sul web).

---

## A — `label_A_merge_random.csv` (100 righe)

Ogni riga è un evento del dataset i cui record hanno nomi diversi tra loro. La domanda: **tutti i nomi indicano la stessa organizzazione?**

| etichetta | quando |
|---|---|
| `same` | stessa organizzazione, anche se scritta in modo diverso (`ACME.COM` / `Acme Inc` / `acme.com - 50GB leaked`), oppure un nome e un segnaposto (`FILES`, `$43.6M\USA\`) |
| `different` | almeno un nome indica un'organizzazione diversa dalle altre |
| `unsure` | non si può stabilire |

Usare anche la colonna `domains`. Una ricerca sul web è necessaria solo nei casi dubbi.

## B — `label_B_duplicates.csv` (100 righe)

Ogni riga è una coppia di eventi **separati** nel dataset: stessa gang, al massimo 7 giorni di distanza, nomi simili o stesso dominio. La domanda: **sono la stessa rivendicazione?**

| etichetta | quando |
|---|---|
| `same_attack` | stessa vittima, stessa rivendicazione pubblicata due volte o con titoli diversi: andrebbero uniti |
| `same_victim_new_post` | stessa vittima, ma un post distinto e legittimo (`Part 1` / `Part 2`, nuovo pacchetto di dati, ultimatum): è giusto tenerli separati |
| `different` | vittime diverse (`ST Asset Management` / `Summit Asset Management`) o segnaposto non correlati (`Views: 23` / `Views: 30`) |
| `unsure` | non si può stabilire |

`gap_days` sono i giorni tra i due eventi.

## C — `label_C_imputed.csv` (270 righe)

Ogni riga è un valore di **paese** (`attribute = country`, codice ISO a due lettere, es. `US`, `IT`) o di **settore** (`attribute = sector`), che il dataset ha **dedotto**: nessuna fonte lo riportava. La domanda: **il valore è corretto?**

Paese = paese della **sede principale** dell'organizzazione. Per una filiale con sito proprio (`acme.de`), il paese della filiale.

Settore = attività principale, una delle 13 classi:

| classe | include |
|---|---|
| Professional Services | studi legali e contabili, consulenza, ingegneria, architettura, immobiliare, agenzie per il lavoro |
| Manufacturing | industria, costruzioni, materiali da costruzione |
| Technology | software, servizi IT, telecomunicazioni |
| Healthcare | ospedali, cliniche, studi medici, farmaceutica |
| Retail & E-Commerce | commercio al dettaglio e all'ingrosso, e-commerce |
| Financial Services | banche, assicurazioni, investimenti |
| Transportation | logistica, trasporti, spedizioni, automotive |
| Education | scuole, università, formazione |
| Government & Defense | enti pubblici, comuni, forze armate |
| Agriculture and Food Production | agricoltura, alimentare, bevande |
| Energy & Utilities | energia, acqua, gas, rifiuti |
| Hospitality | hotel, ristoranti, turismo, viaggi |
| Other | non profit, enti religiosi, media, sport, altro |

**Come verificare:** sito ufficiale (anche su archive.org), registri delle imprese, LinkedIn, Wikipedia, notizie.
**Non usare** i tracker ransomware: ransomware.live, ransomfeed, ransomlook, ransomwatch, hackmanac, falconfeeds, e simili. Sono le fonti stesse del dataset.

| etichetta | quando |
|---|---|
| `correct` | il valore è giusto |
| `wrong` | il valore è sbagliato: scrivere in `note` il valore corretto |
| `unsure` | organizzazione non identificabile, oppure il settore sta a cavallo di due classi |

Il file non mostra quale metodo ha prodotto il valore, per non influenzare il giudizio. La chiave è in `label_C_key.csv`: non aprirla prima di aver finito.

---

Al termine, `python3 evaluate_labels.py` calcola i tassi d'errore con intervalli di confidenza e l'accordo tra annotatori.
