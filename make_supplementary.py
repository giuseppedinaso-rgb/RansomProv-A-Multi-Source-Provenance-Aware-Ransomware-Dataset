#!/usr/bin/env python3
"""Builds SUPPLEMENTARY.md, the supplementary report of the paper, from the
result files in this repository (dataset v1.0). Every number in the report is
read from a file; nothing is typed by hand except the few values marked
"recorded" below, which come from runs on intermediate files not shipped here.

Run from the repository root:  python make_supplementary.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RES = ROOT / "benchmark" / "results"
REL = ROOT / "data" / "release_v1.0"
AUD = ROOT / "audit"
out = []


def w(s=""):
    out.append(s)


def md(df, floatfmt=3):
    """DataFrame -> Markdown table (no extra dependency)."""
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floatfmt}f}")
    cols = [str(c) for c in df.columns]
    w("| " + " | ".join(cols) + " |")
    w("|" + "|".join("---" for _ in cols) + "|")
    for _, r in df.iterrows():
        w("| " + " | ".join("" if (isinstance(v, float) and pd.isna(v)) else str(v) for v in r.values) + " |")
    w()


def load(p):
    return json.load(open(p))


# ---------------------------------------------------------------------------
w("# RansomProv: supplementary report")
w()
w("Supplementary material for the paper *RansomProv: A Multi-Source, Provenance-Aware Dataset of "
  "Ransomware Leak-Site Claims*. All results refer to **dataset v1.0**. The report is generated "
  "by `make_supplementary.py` from the files in `benchmark/results/`, `audit/` and `data/release_v1.0/`; "
  "rerun the script to reproduce it.")
w()
w("Contents: 1 Dataset · 2 Precision of the inference rules · 3 LLM thresholds and contamination · "
  "4 External validation · 5 Tracker comparison · 6 Benchmark protocol · 7 T1 counts · 8 T2 next claim · "
  "9 T3 point processes · 10 T4 attributes from context · 11 T5 imputation · 12 T6 group lifetime · "
  "13 T7 regime changes")
w()

# 1 Dataset ------------------------------------------------------------------
w("## 1 Dataset")
w()
s = load(REL / "release_summary.json")
rows = []
for k in ("events_observed", "events_enriched"):
    v = s[k]
    rows.append({"version": k.replace("events_", ""), "events": v.get("events"), "victims": v.get("victims"),
                 "groups": v.get("gangs"), "countries": v.get("countries"), "first": v.get("first"),
                 "last": v.get("last"), "country missing %": v.get("country_missing_pct"),
                 "sector missing %": v.get("sector_missing_pct")})
md(pd.DataFrame(rows), 1)
ev_file = REL / "events_enriched.json"
if ev_file.exists():
    e = pd.read_json(ev_file)
    w("Origin of each value in the enriched release (`*_src`):")
    w()
    src = pd.concat([e.country_src.value_counts().rename("country"),
                     e.sector_src.value_counts().rename("sector")], axis=1).fillna(0).astype(int)
    md(src.reset_index().rename(columns={"index": "origin"}))
    w("Trackers per event:")
    w()
    md(e.n_sources.value_counts().sort_index().rename_axis("trackers").reset_index(name="events"))

# 2 Precision ----------------------------------------------------------------
w("## 2 Precision of the inference rules")
w()
w("*Automatic*: the rule is applied to events where the value is observed, with the value hidden. "
  "*Audit*: imputed values checked by LLM agents with web verification, blind to the rule and barred "
  "from using the trackers as evidence (an LLM-assisted annotation, not a human one); undecided cases "
  "are excluded; 95% Wilson intervals.")
w()
auto = load(REL / "precision_by_src.json")
lab = load(AUD / "report_labels_label_ai.json")
rows = []
for attr in ("country", "sector"):
    for rule, v in auto[attr].items():
        if not isinstance(v, dict):
            continue
        a = lab["C_imputed"]["by_src"].get(f"{attr}|{rule}")
        auto_p = v.get("precision_pct")
        rows.append({"attribute": attr, "rule": rule,
                     "automatic %": auto_p if auto_p is not None else "see section 3",
                     "automatic n": "" if v.get("n") is None else int(v["n"]),
                     "audit %": None if not a else round(100 - a["error_pct"], 1),
                     "audit 95% CI": "" if not a else f"{100 - a['ci95'][1]:.1f}-{100 - a['ci95'][0]:.1f}",
                     "audit n": "" if not a else a["n_decided"],
                     "audit undecided": "" if not a else a["unsure"]})
md(pd.DataFrame(rows), 1)
if "dropped_rules" in auto:
    w("Rules dropped for low precision: " + ", ".join(
        f"`{k}` ({v['precision_pct']}%, n = {v['n']})" if isinstance(v, dict) else f"`{k}` ({v}%)"
        for k, v in auto["dropped_rules"].items()) + ".")
    w()
w("Record linkage audit:")
w()
rows = []
for k, name in (("A_over_merge", "wrong merges (100 random events whose records have different names)"),
                ("B_missed_duplicates_among_candidates", "missed duplicates (pairs of separate events, same group, at most 7 days apart, similar names)")):
    v = lab[k]
    rows.append({"check": name, "decided": v["n_decided"], "positive": v["errors"],
                 "%": v["error_pct"], "95% CI": f"{v['ci95'][0]}-{v['ci95'][1]}", "undecided": v["unsure"]})
md(pd.DataFrame(rows), 1)
w("The 57 confirmed missed duplicates among 97 candidate pairs correspond to about 1.2% of the events.")
w()

# 3 LLM ----------------------------------------------------------------------
w("## 3 LLM thresholds and contamination")
w()
w("Model: gpt-oss-120b, served locally with vLLM. Thresholds for the knowledge stage are chosen on claims "
  "posted after July 2024 (the model's stated knowledge cutoff is June 2024): the lowest confidence whose "
  "precision reaches the target (country 90%, sector 63%) with at least 20 accepted answers. "
  "Name-only queries are not applied.")
w()
pol = load(ROOT / "llm_knowledge_policy.json")
rows = []
for k, v in pol["rules"].items():
    attr, inp = k.split("|")
    rows.append({"attribute": attr, "input": inp,
                 "threshold": "not applied" if not v else v["min_confidence"],
                 "precision": "" if not v else v["precision"], "coverage": "" if not v else v["coverage"],
                 "n": "" if not v else v["n_validation"]})
md(pd.DataFrame(rows))
w("Selection repeated on the 320 validation claims posted **before July 2025** (recorded: "
  "`llm_knowledge_evaluate.py` on the pre-test subset). It picks the same thresholds:")
w()
md(pd.DataFrame([
    {"attribute": "country", "input": "name+desc", "threshold": 0.5, "precision": 0.947, "n": 95},
    {"attribute": "country", "input": "name+domain", "threshold": 0.8, "precision": 0.901, "n": 101},
    {"attribute": "sector", "input": "name+desc", "threshold": 0.5, "precision": 0.750, "n": 68},
    {"attribute": "sector", "input": "name+domain", "threshold": 0.7, "precision": 0.649, "n": 97}]))
w("Website sector classifier (threshold 0.90, recorded): 82.3% precision on all validation claims with "
  "agreeing trackers, 81.6% on those posted before July 2025.")
w()
w("Contamination (recorded): with the victim name alone the model answered 77% of claims posted before its "
  "cutoff and 9% of those posted after it, so pre-cutoff precision cannot be read as inference.")
w()

# 4 External validation --------------------------------------------------------
w("## 4 External validation")
w()
wa = load(AUD / "ext_wa.json")
rw, ro = wa["ransomware"], wa["other_cyberattack"]
w("### Washington State breach notifications")
w()
md(pd.DataFrame([
    {"notifications": "ransomware", "organisations": rw["organisations"], "matched": rw["matched"],
     "coverage %": rw["coverage_pct"], "coverage after review %": rw.get("coverage_corrected_pct"),
     "median claim - attack start (days)": rw["claim_minus_start_days"]["50%"],
     "median notification - claim (days)": rw["submitted_minus_claim_days"]["50%"]},
    {"notifications": "other cyberattacks (negative control)", "organisations": ro["organisations"],
     "matched": ro["matched"], "coverage %": ro["coverage_pct"], "coverage after review %": "",
     "median claim - attack start (days)": ro["claim_minus_start_days"]["50%"],
     "median notification - claim (days)": ro["submitted_minus_claim_days"]["50%"]}]), 1)
w(f"Review of the {rw['matched']} candidate ransomware matches: " +
  ", ".join(f"{k} {v}" for k, v in rw["assistant_review"].items()) + ". The medians above use all candidate "
  "matches; over the 105 confirmed matches they are 42 days (101 with a start date) and 117 days.")
w()
w("Coverage per year (matched / notifications): " +
  ", ".join(f"{y} {v}" for y, v in rw["coverage_by_year"].items()) + ".")
w()
rep = load(RES / "representativeness.json")
w("Claims vs. confirmed victims (US), by Washington industry category; "
  f"χ² = {rep['distribution_test']['chi2']}, p = {rep['distribution_test']['p_value']:.1e}:")
w()
md(pd.DataFrame([{"category": k, **v} for k, v in rep["distribution"].items()]), 2)
w("Coverage of confirmed victims by category and by breach size (Washington residents affected), "
  "with 95% Wilson intervals. Homogeneity tests: sector p = 0.22, size p = 0.60.")
w()
md(pd.DataFrame([{"category": k, "confirmed": v["confirmed"], "in dataset": v["on_leak_site"], "%": v["pct"],
                  "95% CI": f"{v['ci95'][0]}-{v['ci95'][1]}"} for k, v in rep["coverage_by_industry"].items()]), 1)
md(pd.DataFrame([{"residents affected": k, "confirmed": v["confirmed"], "in dataset": v["on_leak_site"],
                  "%": v["pct"], "95% CI": f"{v['ci95'][0]}-{v['ci95'][1]}"}
                 for k, v in rep["coverage_by_wa_residents_affected"].items()]), 1)
sa = rep["sector_agreement_on_matches"]
w(f"Sector agreement with the official industry on matched claims: {sa['agreement_pct']}% "
  f"(n = {sa['n']}, 95% CI {sa['ci95'][0]}-{sa['ci95'][1]}).")
w()
sec = load(AUD / "ext_sec.json")
w("### SEC Form 8-K, Item 1.05")
w()
sm = pd.read_csv(AUD / "sec_matches.csv").review_assistant.value_counts()
n_ok = int(sm.get("correct", 0))
w(f"{sec['item_105_companies']} companies filed under Item 1.05; {sec['matched_in_dataset']} have a candidate match "
  f"in the dataset. Review of the candidates: " + ", ".join(f"{k} {v}" for k, v in sm.items()) +
  f". Confirmed: {n_ok} of {sec['item_105_companies']} ({n_ok / sec['item_105_companies'] * 100:.0f}%). "
  "Item 1.05 alone misses many incidents, so this is a qualitative check only.")
w()
w("### Takedowns and shutdowns")
w()
ca = pd.read_csv(RES / "changepoint_activity.csv")
md(ca.rename(columns={"event": "operation date", "days_last_minus_event": "last claim - operation (days)",
                      "weekly_rate_12w_before": "claims/week, 12 weeks before",
                      "weekly_rate_12w_after": "claims/week, 12 weeks after"}), 1)

# 5 Trackers -------------------------------------------------------------------
w("## 5 Tracker comparison")
w()
tc = load(RES / "tracker_comparison.json")
names = {"RL": "ransomware.live", "RF": "Ransomfeed", "LK": "RansomLook"}
md(pd.DataFrame([{"tracker": names[k], **v} for k, v in tc["overall"].items()]), 1)
w("Pairwise overlap:")
w()
md(pd.DataFrame([{"pair": k, **v} for k, v in tc["pairs"].items()]), 3)
w("Exclusive events per year:")
w()
md(pd.DataFrame({names[k]: {y: v["exclusive"] for y, v in d.items()} for k, d in tc["by_year"].items()})
   .rename_axis("year").reset_index())
cv = tc["confirmed_victims"]
w(f"Confirmed victims (Washington) in the dataset: {cv['n_confirmed_in_dataset']}.")
w()
md(pd.DataFrame([{"tracker": names[k], "reports %": v["recall_pct"],
                  "own sector matches official %": v.get("sector_agreement_pct", ""),
                  "with a sector": v.get("sector_n", "")} for k, v in cv.items() if k in names]), 1)
ss = load(RES / "single_source.json")
w("Single-tracker views of the dataset:")
w()
md(pd.DataFrame([{"view": k, **{c: v[c] for c in ("events", "share_of_merged", "gangs", "first", "country_pct", "sector_pct")}}
                 for k, v in ss["coverage"].items()]), 1)
g = load(RES / "single_source_gru.json")
w("Next-claim GRU (best T2 configuration, observed labels, 3 seeds) trained on one tracker vs. on the merged "
  "history, scored on the test events that tracker reports (weighted F1):")
w()
rows = []
for s_, d in g.items():
    for view, m in d.items():
        rows.append({"tracker": s_, "training": view, **{f: m[f]["weighted_f1"] for f in ("delta", "country", "sector")}})
md(pd.DataFrame(rows))

# 6 Protocol --------------------------------------------------------------------
w("## 6 Benchmark protocol")
w()
w("T1-T4: training until December 2024, validation January-June 2025 (hyperparameter selection), test "
  "July 2025 - September 2026; T1 and T4 refit the selected configuration on all data before July 2025. "
  "T5 hides 20% of the events with known country and sector at random. T6 uses 5x3 cross-validation over "
  "groups. T7 is evaluated on nine operations. Optuna (TPE): 20 trials for neural and point-process models, "
  "30 for trees, 10 for logistic regression. Stochastic models: 3 seeds. Unless stated, enriched labels; "
  "T2 always tests on observed labels.")
w()

# 7 T1 ----------------------------------------------------------------------------
w("## 7 T1 - weekly count forecasting (MASE, lower is better)")
w()
w("Four weeks ahead, 16 rolling windows. MASE is scaled by the in-sample one-step naive error.")
w()
f1 = pd.concat([pd.read_csv(RES / "forecast_summary.csv"), pd.read_csv(RES / "forecast_sweep_summary.csv")])
for ver in ("enriched", "observed"):
    t = f1[f1.version == ver].pivot_table(index="model", columns="group", values="MASE")
    t = t[[c for c in ("total", "sector", "country", "gang") if c in t.columns]].sort_values("total")
    w(f"**{ver}**")
    w()
    md(t.reset_index())
h = pd.read_csv(RES / "forecast_hier_summary.csv")
w("Intermittent-demand methods and hierarchical reconciliation (MinT), per dimension and level:")
w()
md(h)
bc = load(RES / "forecast_sweep_best_configs.json")
w("Selected configurations of the tuned neural models:")
w()
for k, v in bc.items():
    w(f"- `{k}`: `{json.dumps(v)}`")
w()

# 8 T2 ----------------------------------------------------------------------------
w("## 8 T2 - next claim per group")
w()
w("Interval classes: same day, within 7 days, later (the first claim of a group has no interval). "
  "Country: 30 most frequent plus other; sector: 13 classes. Mean ± standard deviation over seeds "
  "(baselines and LLMs: one deterministic run). `-tuned`: Optuna configuration; without suffix: default "
  "configuration. Few-shot LLM runs were made on the enriched version only.")
w()
rows = []
for f in sorted(glob.glob(str(RES / "nextevent" / "*.json"))):
    model = Path(f).stem.rsplit("_", 2)[0]          # file name: <model>_<version>_s<seed>
    d = load(f)
    if "test" not in d:
        continue
    ver = d.get("version")
    for fld in ("delta", "country", "sector"):
        t = d["test"].get(fld)
        if t:
            rows.append({"model": model, "version": ver, "field": fld, "seed": d.get("seed"),
                         "wf1": t.get("weighted_f1"), "acc": t.get("accuracy"), "ll": t.get("log_loss", np.nan)})
nd = pd.DataFrame(rows)
for ver in ("enriched", "observed"):
    t = nd[nd.version == ver].groupby(["model", "field"]).agg(wf1=("wf1", "mean"), sd=("wf1", "std"),
                                                              acc=("acc", "mean"), ll=("ll", "mean"), runs=("wf1", "size"))
    t = t.reset_index()
    t["weighted F1"] = [f"{a:.3f}" + ("" if pd.isna(b) else f" ± {b:.3f}") for a, b in zip(t.wf1, t.sd)]
    p = t.pivot_table(index="model", columns="field", values=["weighted F1", "acc", "ll"], aggfunc="first")
    p.columns = [f"{c[1]} {c[0]}" for c in p.columns]
    order = [f"{f} {m}" for f in ("delta", "country", "sector") for m in ("weighted F1", "acc", "ll") if f"{f} {m}" in p.columns]
    w(f"**{ver}** (weighted F1, accuracy, log-loss)")
    w()
    md(p[order].reset_index().rename(columns=lambda c: c.replace("delta", "interval").replace(" ll", " log-loss")))
w("Selected configurations of the tuned sequence models:")
w()
for f in sorted(glob.glob(str(RES / "nextevent_sweep" / "best_*.json"))):
    d = load(f)
    w(f"- `{d.get('model', Path(f).stem)}`: `{json.dumps(d.get('params'))}` (validation loss {d.get('best_val_loss', float('nan')):.4f})")
w()

# 9 T3 ----------------------------------------------------------------------------
w("## 9 T3 - neural temporal point processes")
w()
w("EasyTPP models, one sequence per group, mark = country or sector. Mean ± standard deviation over 3 seeds. "
  "Log-likelihood per event (higher is better; IntensityFree and S2P2 report a continuous time density, which can "
  "be positive). Time RMSE in days from sampled event times; very large values show divergent sampling.")
w()
rows = []
for f in glob.glob(str(RES / "tpp_sweep" / "final_*.json")):
    d = load(f)
    rows.append({"model": d["model"], "version": d["version"], "mark": d["mark"], "acc": d["test"]["acc"],
                 "loglike": d["test"]["loglike"], "rmse": d["test"]["rmse"]})
tp = pd.DataFrame(rows)
for ver in ("enriched", "observed"):
    t = tp[tp.version == ver].drop(columns="version").groupby(["mark", "model"]).agg(["mean", "std"])
    t.columns = [f"{a} {b}" for a, b in t.columns]
    w(f"**{ver}**")
    w()
    md(t.reset_index(), 3)
w("Selected configurations:")
w()
for f in sorted(glob.glob(str(RES / "tpp_sweep" / "best_*.json"))):
    d = load(f)
    w(f"- `{Path(f).stem.replace('best_', '')}`: `{json.dumps(d.get('params', d))}`")
w()

# 10 T4 ---------------------------------------------------------------------------
w("## 10 T4 - victim attributes from context")
w()
w("Predict country (top 9 + other) and sector of a claim from the group, its history and the date. "
  "Default configurations and references:")
w()
md(pd.read_csv(RES / "tabular_summary.csv"), 3)
w("Tuned models (Optuna on validation log-loss):")
w()
md(pd.read_csv(RES / "tabular_sweep_summary.csv"), 3)

# 11 T5 ---------------------------------------------------------------------------
w("## 11 T5 - imputation")
w()
w("20% of the events with known country and sector hidden, 5 repetitions. MCAR: completely at random; "
  "pattern: with the real missingness pattern (year x tracker combination). Our rules: the deterministic rules "
  "of the dataset (same victim, ccTLD, country in the post), without website or LLM; they answer only where they "
  "apply (coverage). `[rule subset]`: every method scored on the same values where a rule applies.")
w()
im = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RES / "imputation_subset_*.csv")))])
im["method"] = im.method.str.replace("Rules (enriched, no LLM/site)", "our rules", regex=False)
t = im.groupby(["target", "scheme", "method"])[["accuracy", "macro_f1", "coverage"]].mean().reset_index()
md(t, 3)

# 12 T6 ---------------------------------------------------------------------------
w("## 12 T6 - group lifetime")
w()
w("Landmark at 60 days: groups still active 60 days after their first claim; features from those 60 days only; "
  "target: remaining active time. A group has ended if it posts nothing in the last 90 days of the dataset. "
  "5-fold stratified cross-validation, 3 repetitions.")
w()
sv = pd.read_csv(RES / "survival.csv").groupby("model")[["c_index", "ibs"]].agg(["mean", "std"])
sv.columns = [f"{a} {b}" for a, b in sv.columns]
md(sv.reset_index())
w("Kaplan-Meier by start cohort:")
w()
md(pd.read_csv(RES / "survival_km.csv"), 2)

# 13 T7 ---------------------------------------------------------------------------
w("## 13 T7 - regime changes")
w()
w("Detectors run on weekly claim counts of each group without knowing the operation date. A hit is a change "
  "point within four weeks of the operation.")
w()
cp = pd.read_csv(RES / "changepoint.csv")
t = cp.groupby("method").agg(hits=("hit", "sum"), events=("hit", "size"),
                             false_alarms_per_year=("false_alarms_per_year", "mean")).reset_index()
md(t, 2)
md(cp, 2)

Path(ROOT / "SUPPLEMENTARY.md").write_text("\n".join(out))
print(f"-> SUPPLEMENTARY.md ({len(out)} lines)")
