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

## Why the event files are not here yet

The events are derived from three trackers with different terms:

| tracker | terms |
|---|---|
| RansomLook | CC BY 4.0: redistribution with attribution |
| ransomware.live | non-commercial academic use with attribution; redistribution of raw data requires written permission |
| Ransomfeed | no machine-readable terms |

Permission has been requested from ransomware.live and Ransomfeed. The event files will be published here and on Zenodo (with a DOI) once it is granted. Until then, the dataset can be rebuilt with the code in this repository and your own access to the trackers.
