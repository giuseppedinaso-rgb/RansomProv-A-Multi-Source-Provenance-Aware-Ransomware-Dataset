# RansomProv 
Harmonised Ransomware Leak-Site Claims (2018–2026)

Code, documentation and benchmark for a dataset of ransomware leak-site claims merged from three public trackers: [ransomware.live](https://www.ransomware.live), [Ransomfeed](https://ransomfeed.it) and [RansomLook](https://www.ransomlook.io). Every country and sector value carries its provenance, and the precision of every inference rule is measured.

| | v1.0 (paper) | v1.1 (maintenance) |
|---|---|---|
| events | 37,385 | 36,810 |
| distinct victims | 33,148 | 32,738 |
| groups | 410 | 410 |
| period | 2018-01-03 – 2026-09-30 | same |
| country missing, observed → enriched | 24.9% → 13.6% | 24.3% → 13.2% |
| sector missing, observed → enriched | 14.9% → 5.0% | 14.3% → 4.6% |

All results in the paper use **v1.0**. v1.1 applies the findings of the audit; see `data/release_v1.1/CHANGELOG.md`.

> **Data availability.** The event files are not in this repository yet. RansomLook data is CC BY 4.0. Ransomware.live forbids redistribution of its raw data without written permission, and Ransomfeed does not publish machine-readable terms; permission from both has been requested. Until it is granted, the repository contains the code, the documentation, the crosswalks, the measured precision, the audit labels and all benchmark results, and the dataset can be rebuilt with your own access to the trackers. See `data/README.md`.

## Repository layout

| path | content |
|---|---|
| `fetch_*.py` | stage 1: download all records from the three tracker APIs (query partitioning where an API caps results) |
| `build_sector_crosswalk.py`, `sector_crosswalk.json` | stage 2: map Ransomfeed's 175 sector labels to the 13 ransomware.live classes |
| `normalize.py`, `ransomlook_clean.py`, `gang_aliases_manual.json` | normalisation of countries, victim names, domains and group names |
| `stage3_unify.py` | stage 3: one record per row, ISO countries, canonical groups, victim keys |
| `stage4_events.py` | stage 4: record linkage into events (union-find on group + victim key within a time window), conflict resolution, deterministic rules |
| `site_enrich.py`, `site_sector.py`, `stage5c_merge_site.py` | stage 5c–5d: country from the last archive.org snapshot before the claim; sector from the page text with a local LLM |
| `llm_knowledge.py`, `llm_knowledge_evaluate.py`, `llm_knowledge_policy.json`, `stage5e_merge_knowledge.py` | stage 5e: country and sector from model knowledge, thresholds selected only on post-cutoff claims |
| `make_release.py` | builds the observed and enriched releases |
| `validate_rules.py` | automatic precision of each rule (value hidden on events where it is observed) |
| `audit_dedup.py`, `make_label_samples.py`, `merge_ai_labels.py`, `evaluate_labels.py`, `audit/` | audit of merges, missed duplicates and imputed values |
| `ext_wa.py`, `ext_sec.py`, `ext_match.py` | external validation: Washington State breach notifications, SEC 8-K Item 1.05 |
| `analysis_sources.py` | per-tracker statistics |
| `benchmark/` | the seven benchmark tasks (T1–T7) and the tracker analyses |
| `benchmark/results/` | all benchmark outputs (summaries and per-run files) |
| `build_dataset_2026.ipynb` | notebook describing every stage |
| `data/` | release folders: README, datasheet, changelog, crosswalk to NAICS/ISIC, measured precision |
| `release` | link to `data/release_v1.0`, the version read by the benchmark scripts |

Code comments and the notebook are in Italian.

## Rebuilding the dataset

Python 3.10. Install `requirements-pipeline.txt`. Run from the repository root.

```bash
export RANSOMLIVE_API_KEY=...          # ransomware.live PRO API key
python fetch_ransomfeed.py
python fetch_ransomlive.py
python fetch_ransomlook.py             # -> raw/
python build_sector_crosswalk.py
python stage3_unify.py                 # -> build/unified.json
python stage4_events.py                # -> build/events.json (EVENT_WINDOW, default 7 days in v1.1, 3 in v1.0)
```

Stages 5c–5e need an OpenAI-compatible server with `gpt-oss-120b` (we used vLLM, tensor parallel on 4 GPUs) at `LOCAL_LLM_URL`:

```bash
python site_enrich.py validate && python site_enrich.py apply
python site_sector.py validate && python site_sector.py apply
python stage5c_merge_site.py
python llm_knowledge.py validate && python llm_knowledge_evaluate.py && python llm_knowledge.py apply
python stage5e_merge_knowledge.py
RELEASE_DIR=release_v1.1 python make_release.py
```

The current code produces v1.1. v1.0 used a 3-day window and the earlier versions of `normalize.py`, `ransomlook_clean.py` and `stage4_events.py`; the differences are listed in the v1.1 changelog.

## Benchmark

Install `requirements-benchmark.txt`. All tasks share one temporal split: training until December 2024, validation January–June 2025, test July 2025 – September 2026. Hyperparameters are chosen on validation only (Optuna); stochastic models run with 3 seeds.

| task | scripts |
|---|---|
| T1 weekly count forecasting | `forecast.py`, `forecast_sweep.py`, `forecast_hier.py` |
| T2 next claim per group | `nextevent_data.py`, `nextevent_baselines.py`, `nextevent_ptls.py`, `nextevent_sweep.py`, `nextevent_flextpp.py`, `nextevent_llm.py` |
| T3 temporal point processes | `tpp_prepare.py`, `tpp_run.py`, `tpp_sweep.py` |
| T4 victim attributes from context | `tabular.py`, `tabular_sweep.py` |
| T5 imputation | `imputation.py` |
| T6 group lifetime | `survival.py` |
| T7 regime changes | `changepoint.py` |
| tracker analyses | `tracker_comparison.py`, `single_source.py`, `single_source_gru.py`, `representativeness.py` |

Some scripts also read intermediate files from `build/` (record-to-event map, victim keys, raw Washington notifications), which the pipeline regenerates.

## Ethics

The dataset contains names of organisations that ransomware groups published to pressure them. It contains no stolen data, no leaked files and no links to leak sites. Claims are not verified and can be false or recycled. Archived snapshots were fetched only for public pages and only before the claim date. Use the data for aggregate research and defence, not to identify or contact victims.

## Licence

Code: MIT (see `LICENSE`). Data: see `data/README.md`.
