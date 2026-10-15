# Data

Two releases, each with its own README (datasheet), changelog and summary:

| folder | content |
|---|---|
| `release_v1.0/` | version used for every result in the paper |
| `release_v1.1/` | maintenance release with the audit findings applied |

Each release contains:

| file | in this repository | content |
|---|---|---|
| `events_observed.{json,csv}` | **not yet** | country and sector only when reported by a tracker |
| `events_enriched.{json,csv}` | **not yet** | all values, including inferences, with the origin of each value (`*_src`) |
| `README.md` | yes | datasheet: fields, construction, known limits |
| `CHANGELOG.md` | yes | changes between versions |
| `release_summary.json` | yes | counts and missing rates |
| `precision_by_src.json` | yes | measured precision of every inference rule |
| `sector_crosswalk_naics_isic.csv` | yes | alignment of the 13 sector classes with NAICS 2022 and ISIC Rev. 4 |
| `figures/` | yes | tracker overlap |

## Public release

The event files will be published upon publication of the paper, here and on Zenodo (with a DOI).

| | internal file | public release |
|---|---|---|
| enriched, v1.0 | 29 columns | 27 columns: `gang_raw` and `victim_raw` removed |
| observed, v1.0 | 21 columns | 19 columns |
| v1.1 | one more column, `not_a_victim` | same removals |

`gang_raw` and `victim_raw` hold the strings as posted by the trackers. All other columns hold harmonised values: canonical group names, cleaned victim names, ISO country codes, the 13 sector classes, provenance tags and enrichment metadata. Each event lists the trackers that report it (`sources`), so the original records can be consulted under each tracker's terms:

| tracker | terms |
|---|---|
| RansomLook | CC BY 4.0: redistribution with attribution |
| ransomware.live | non-commercial academic use with attribution; redistribution of raw data requires written permission |
| Ransomfeed | no machine-readable terms |
