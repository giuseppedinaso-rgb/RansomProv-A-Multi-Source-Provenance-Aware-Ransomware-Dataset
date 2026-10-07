# Harmonised Ransomware Leak-Site Events (2018–2026) — v1.0

Version used for the paper's benchmarks. A corrected v1.1 is available (see CHANGELOG.md).

Event-level dataset of ransomware attacks claimed on data-leak sites, built by harmonising three public trackers. Each row is one **event**: one victim claimed by one group, with posts from different trackers merged when they refer to the same claim.

Two versions are released from the same master file:

| file | content |
|---|---|
| `events_observed.{json,csv}` | country and sector **only when reported by at least one source for that event**; every inferred value is null |
| `events_enriched.{json,csv}` | all values, including inferences; the `*_src` columns state where each value comes from |

`precision_by_src.json` gives the measured precision of every inference rule. `release_summary.json` gives the counts below.

## Summary

| | observed | enriched |
|---|---|---|
| events | 37,385 | 37,385 |
| distinct victims | 33,148 | 33,148 |
| groups (gangs) | 410 | 410 |
| countries | 183 | 191 |
| sectors | 13 | 13 |
| period | 2018-01-03 – 2026-09-30 | same |
| country missing | 24.9% | **13.6%** |
| sector missing | 14.9% | **5.0%** |

Events per source combination: all three trackers 22,103; two trackers 7,161; one tracker 8,121 (4,622 RansomLook only).

## Sources

| source | records retrieved | fields used |
|---|---|---|
| ransomware.live (API PRO) | 32,111 | victim, website, country, activity (sector), description, group, dates |
| ransomfeed.it (API) | 25,675 | victim, website, country, work sector, group, date |
| RansomLook (`/api/posts`) | 35,256 | post title (victim), group, date |

Records were retrieved on 1 October 2026 through each tracker's API, with query partitioning wherever an endpoint caps the number of results (details in the pipeline notebook).

## Fields

| field | description |
|---|---|
| `event_id` | integer identifier, stable within this release |
| `discovered` | date the claim was first observed (YYYY-MM-DD) |
| `victim` | cleaned victim name: HTML entities decoded, URL reduced to host, leak-site tags and tails such as "1 TB DATA LEAKED" removed |
| `victim_raw` | name as published by the source |
| `victim_truncated` | the leak site truncated the name ("acme-indus...") |
| `victim_placeholder` | the post title does not name a victim ("FILES", "ARCHIVE6", "Views: 23"); the claim is kept |
| `domain` | victim website domain; null when the source value is not a valid domain or is a third-party aggregator (e.g. zoominfo.com) |
| `gang` | canonical group name (aliases merged; see `gang_aliases_manual.json`) |
| `gang_raw` | group name as published |
| `country` | ISO 3166-1 alpha-2 country of the victim |
| `country_src` | origin of `country` (see below) |
| `country_resolution` | for observed values: `unanimous`, or how a conflict between sources was resolved (`cctld`, `description`, `source_priority`) |
| `country_conflict`, `country_candidates` | whether the sources disagreed, and their values |
| `sector` | one of 13 classes (ransomware.live taxonomy): Professional Services, Manufacturing, Technology, Healthcare, Retail & E-Commerce, Financial Services, Transportation, Education, Government & Defense, Agriculture and Food Production, Energy & Utilities, Hospitality, Other |
| `sector_src`, `sector_resolution`, `sector_conflict`, `sector_candidates` | as for country |
| `sources`, `n_sources` | trackers that reported the event |

Enriched version only:

| field | description |
|---|---|
| `site_rule`, `site_snapshot` | for `site_*` countries: decision rule and archive.org snapshot timestamp (always earlier than the attack) |
| `sector_model`, `sector_confidence` | for `site_llm` sectors |
| `knowledge_input`, `knowledge_model`, `knowledge_confidence_country`, `knowledge_confidence_sector` | for `llm_knowledge` values |

## Provenance of country and sector

| `*_src` | meaning | precision (validation) |
|---|---|---|
| `observed` | reported by at least one source for this event | — |
| `cctld` | country-code top-level domain of the victim's website | country 99.3% (n=7,607) |
| `description` | exactly one country named in the leak-post description | country 95.4% (n=6,470) |
| `site_archive` / `site_live` | address, VAT/registry number, schema.org data or phone numbers on the victim's website, from an archive.org snapshot taken before the attack, or else from the live site | country 96.0% / 94.3% (n=272 / 35) |
| `inherited_victim` | value observed for the same victim in another event | country 82.7% (n=5,233); sector 63.8% (n=6,297) |
| `site_llm` | sector classified from the website text by gpt-oss-120b (local, confidence ≥ 0.90) | sector 82.3% (n=164) |
| `llm_knowledge` | gpt-oss-120b (local, no web search) given victim name plus leak description or domain | country 93.2% / 90.4%; sector 76.1% / 65.3% (description / domain input) |
| `missing` | no value | — |

Precision is measured by hiding the value on events where it is observed and applying the rule. The CIs and majority-class baselines are in `precision_by_src.json`. These events are not a random sample of the events where the rules were applied, so the figures are estimates.

For reference, the two sources that report a sector agree on the same event in only 59.6% of cases (Cohen's κ = 0.54). Inherited sectors are therefore about as reliable as observed ones.

`llm_knowledge` thresholds were selected only on events discovered **after the model's training cutoff (July 2024)**. Before that date the model may have memorised tracker pages, which are the same sources and would inflate accuracy. Inputs consisting of the victim name alone could not be validated without this contamination and were **not** used.

## Source overlap, timeliness and robustness

- **Overlap.** 59% of events are reported by all three trackers; 4,622 only by RansomLook, 2,959 only by ransomware.live, 540 only by ransomfeed (`figures/overlap.png`).
- **Timeliness.** For events seen by several trackers, the median difference in first-seen date is 0 days for every pair of sources. Within one day: 85.9–98.5% depending on the pair.
- **Unobserved events.** Capture–recapture with log-linear models (2023–2026) puts the union of the three trackers at roughly 87–97% of estimated claims. The estimates are model-dependent and should be read as an order of magnitude.
- **Event window.** Varying the merge window from 1 to 30 days changes the number of events by 4%, and leaves group concentration, US share and coverage within 0.5 points.
- **Sector standards.** `sector_crosswalk_naics_isic.csv` maps the 13 classes to NAICS 2022 and ISIC Rev.4 (approximate, many-to-many).

## External validation

- **Washington State AG breach notifications** (open data, data.wa.gov `sb4j-ca4h`). Of 460 organisations that notified a **ransomware** attack (June 2020 – June 2026), 22.8% match a claim in this dataset. Match precision on manual review is 98% (105 of 107 decided). Negative control: 9.2% of non-ransomware notifications match. Claims appear a median of 43 days after the attack started (IQR 19–61) and **114 days before** the official notification (IQR 32–206).
- **SEC 8-K Item 1.05** (material cybersecurity incidents of US-listed companies, since December 2023): 11 of 58 companies matched (19%; a lower bound, since not every Item 1.05 incident is ransomware).
- Leak-site data therefore covers a minority of confirmed ransomware victims, but anticipates official disclosure by several months.


**Independent check of imputed values (v1.0).** About 30 imputed values per source were verified on the web by LLM agents (Claude, Anthropic). The source of each value was hidden from the agents, and ransomware trackers were not allowed as evidence. This is LLM-assisted annotation, not human annotation. Results (precision, 95% CI): ccTLD 82.8% (65.5–92.4), description 88.0% (70.0–95.8), same-victim inheritance 92.9% (77.4–98.0), LLM knowledge 91.3% (73.2–97.6), website archive 93.3% (78.7–98.2) and live 86.2% (69.4–94.5) for country; same-victim inheritance 57.1% (39.1–73.5), LLM knowledge 64.7% (41.3–82.7), website text 85.2% (67.5–94.1) for sector. The automatic estimate for the ccTLD rule (99.3%) is optimistic: it is measured on events whose country is observed, where ccTLDs behave normally. Among imputed events the errors come from truncated domains, ccTLDs used as state or city codes, and generic ccTLDs. These are corrected in v1.1.
Same check on deduplication: wrong merges 1/100 random multi-name events (1.0%, CI 0.2–5.4); among candidate duplicate pairs, 57/97 were the same claim. This corresponds to about 1.2% residual duplicate events in v1.0.

## Known limitations

- **Country coverage 2021–2022.** Share of events without a country, observed / enriched: 2021 86.4% / 67.1%; 2022 80.5% / 46.8%; from 2023 onwards 10–13% / 4–5%. ransomfeed covers almost only 2023 onwards, ransomware.live reports a country for 13–20% of its 2021–2022 records, and RansomLook has no country field. Per-country time series before 2023 are not reliable.
- **Geographic skew.** 46% of events with a country are in the United States. This reflects what leak sites publish and what trackers capture, not the true distribution of attacks.
- **Leak-site claims, not confirmed incidents.** A claim may be false, recycled from another breach, or concern a subsidiary. Dates are publication or discovery dates, not intrusion dates.
- **Deduplication.** Posts are merged into one event when they concern the same victim (name or domain key) and group within 3 days. Re-posts of the same victim by the same group more than 3 days apart are separate events. Reproducible audit (`audit_dedup.py`): 512 events with dissimilar names among their records, mostly the same victim under different titles; 790 candidate pairs of split duplicates, 503 within 3 days. Random samples of 100 per type, for manual labelling, are in `pipeline2026/audit/`; the labelled error rates will be reported once the samples are annotated.
- **Coverage.** The union of three trackers, not of all leak sites. RansomLook-only events (4,622) have no country or sector from the sources.

## Licensing and terms of the sources

- **RansomLook**: content released under CC BY 4.0; credit "RansomLook (ransomlook.io)".
- **ransomware.live**: free for non-commercial and academic use with attribution ("Source: Ransomware.live"). Its terms forbid commercial use without written approval, redistribution of the data as an API, and **publishing the raw data**.
- **ransomfeed.it**: terms to be confirmed with the maintainers before release.

> **Before public release**, obtain written confirmation from ransomware.live (and ransomfeed) that publishing this derived, event-level dataset is acceptable. If it is not, release only the RansomLook-derived fields plus the pipeline code, with which users can rebuild the dataset using their own API access.

This dataset is intended for **non-commercial research**. Cite all three sources.

## Ethical considerations

- Victims are organisations named publicly by criminal groups to extort them. The dataset reproduces names already public on the trackers. It contains **no leaked data**, no files, no links to leak sites and no leak-post descriptions.
- Some victims are small organisations or individuals' practices. Do not use the dataset to profile, contact or shame victims.
- Inclusion does not mean that an organisation was breached: the claims are unverified.

## Reproducibility

The full pipeline is in `pipeline2026/`, documented in `build_dataset_2026.ipynb`: retrieval, unification, events, website enrichment, LLM enrichment, validation and release. The LLM steps use the public `openai/gpt-oss-120b` weights served locally with vLLM 0.30.0 (temperature 0, reasoning effort low). Prompts and every raw model answer are kept in `build/*.jsonl`.
