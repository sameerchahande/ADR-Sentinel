# ADR-Sentinel v2 - AI-Assisted ADR Detection & Auto-Reporting System

An academic decision-support toolkit that predicts adverse drug reactions (ADRs)
from the drug's ATC class profile, bands them by risk, grades severity, scores
causality with the Naranjo scale, and then **auto-drafts a PvPI-style Individual
Case Safety Report (ICSR)** as a review-ready PDF.

Built with standard scikit-learn estimators fitted to local public reference data.
No cloud API key is required. This prototype predicts reference associations, not
patient-specific ADR incidence.

---

## 1. Problem statement

> India's ADR reporting rate is below 1%, against a global average of 6-10%
> (WHO/PvPI data). Studies attribute this primarily to two barriers:
> (1) low awareness/detection capability among healthcare professionals, and
> (2) time-consuming, friction-heavy manual reporting processes to PvPI/ADR
> Monitoring Centres. Drug-related hospital admissions range from 0.2% to 22%
> globally (WHO), rising to 10-20% in geriatric patients, most of which are
> preventable with earlier detection. This project targets BOTH barriers, not
> just detection.

<!--
India's ADR reporting rate is below 1%, against a global average of 6-10% 
(WHO/PvPI data). Studies attribute this primarily to two barriers: 
(1) low awareness/detection capability among healthcare professionals, and 
(2) time-consuming, friction-heavy manual reporting processes to PvPI/ADR 
Monitoring Centres. Drug-related hospital admissions range from 0.2% to 22% 
globally (WHO), rising to 10-20% in geriatric patients, most of which are 
preventable with earlier detection. This project targets BOTH barriers, not 
just detection.
-->

The problem statement above is supplied project background. Its numerical claims
need primary-source citations and denominator verification before publication;
they were not established by this model evaluation.

---

## 2. How the tool answers the problem

| Barrier | What ADR-Sentinel v2 does | Where |
|---|---|---|
| **1. Low detection capability** | Predicts the drug's ADR profile from its ATC class, bands every reaction as Low / Medium / High, grades severity (mild / moderate / severe / life-threatening) and scores provisional causality (Naranjo; WHO-UMC assessed separately) so a non-specialist gets an interpretable risk read-out | `sentinel/engine.py`, `sentinel/detection.py` (Module 1) |
| **2. Reporting friction** | `generate_icsr_draft()` pre-fills simplified core ICSR fields (patient, suspect drug, reaction, onset, severity, outcome, causality, narrative) as structured JSON **or a formatted PDF**, with per-field provenance and an explicit review checklist | `sentinel/reporting.py` (Module 2) |
| Delivering both at the point of care | One FastAPI service (`/analyze`, `/generate-report`, `/health`, `/demo`) plus a single-page web UI that runs the whole predict to auto-draft flow, containerised with a healthcheck | `sentinel/api.py`, `static/` (Modules 3-4) |

**Proposed reporting-friction mechanism:** identify a suspected *observed* reaction
→ prefill a simplified draft from confirmed case details and reference data →
pharmacist verifies missing fields, causality and seriousness → submit through an
authorised PvPI/AMC channel. A predicted association alone is not an observed ADR
and must not be submitted as a case. A two-minute review is a usability **target**,
not a measured result; improved reporting rates require a prospective workflow study.

---

## 3. Architecture

```
                    SIDER reference data (public files, shipped locally)
   drug_names.tsv ─    drug_atc.tsv ┐    meddra_all_se_tsv.gz ─┐
                   │                 │                          │
                   ▼                 ▼                          ▼
      ┌────────────────────────────────────────────────────────────────────┐
      │ sentinel/engine.py        MODULE 1 - DETECTION ENGINE              │
      │  • ATC multi-hot features (hierarchy levels 1, 3, 4)               │
      │  • candidate models: RandomForest / ExtraTrees / LogReg / ensemble │
      │  • per-label F1-tuned decision thresholds                          │
      │  • artefact cache  ->  models/adr_random_forest.joblib             │
      └───────────────────────────────┬────────────────────────────────────┘
                                      ▼
      ┌────────────────────────────────────────────────────────────────────┐
      │ sentinel/detection.py                                              │
      │  • risk banding   Low <40%  |  Medium 40-70%  |  High >70%         │
      │  • rapidfuzz drug-name matching + symptom matching                 │
      │  • Naranjo scale (10 weighted items; provisional assessment)        │
      │  • local keyword severity suggestions (review required)             │
      └───────────────────────────────┬────────────────────────────────────┘
                                      ▼
      ┌────────────────────────────────────────────────────────────────────┐
      │ sentinel/reporting.py     MODULE 2 - AUTO-REPORTING ASSISTANT      │
      │  generate_icsr_draft()  ->  ICSR dict, provenance per field        │
      │  render_icsr_pdf()      ->  print-ready A4 PDF (reportlab)         │
      └───────────────────────────────┬────────────────────────────────────┘
                                      ▼
      ┌────────────────────────────────────────────────────────────────────┐
      │ sentinel/service.py   shared orchestration for the API and the CLI │
      └───────────────┬───────────────────────────────┬────────────────────┘
                      ▼                               ▼
    ┌─────────────────────────────────┐   ┌─────────────────────────────────┐
    │ sentinel/api.py    MODULE 3     │   │ adr_prediction_model.py         │
    │  POST /analyze                  │   │  interactive CLI + --report     │
    │  POST /generate-report (JSON/PDF)│  └─────────────────────────────────┘
    │  GET  /demo  /health            │
    │  GET  /model-performance        │
    └───────────────┬─────────────────┘
                    ▼
    ┌─────────────────────────────────┐
    │ static/            MODULE 4     │
    │  index.html + app.js + styles   │
    │  predict -> colour-coded table  │
    │  -> "Draft PvPI report" -> PDF  │
    └─────────────────────────────────┘
```

### Repository layout

```
adr_prediction_model.py        CLI entry point (interactive + flags) + problem statement
main.py                        application entry point (python main.py)
run.py                         compatibility entry point
sentinel/                      application package
  config.py                    paths, thresholds, risk bands, env overrides
  engine.py                    SIDER loading, features, training, model selection, inference
  evaluation.py                metrics report (accuracy, F1, AUROC, precision@k, baselines)
  figures.py                   the PNG performance charts embedded in section 4
  detection.py                 risk banding, fuzzy matching, severity grading, Naranjo
  reporting.py                 ICSR draft builder + PDF renderer   (MODULE 2)
  service.py                   orchestration shared by the API and the CLI
  api.py                       FastAPI service                    (MODULE 3)
  build_model.py               train, evaluate, render figures, bake the demo cache
  data/severity_keywords.json  local severity keyword table
  data/naranjo_questions.json  the 10 standard Naranjo items and cut-offs
  data/atc_labels.json         readable ATC level-3 therapeutic labels
static/                        single-page front end               (MODULE 4)
tests/                         pytest suite (detection, reporting, service, API)
models/  reports/  data/       generated artefacts (git-ignored, rebuilt in one command)
Dockerfile  docker-compose.yml  .env.example  requirements*.txt
```

---
## 4. Model performance (measured, not claimed)

Reproduce with `python -m sentinel.build_model` — it writes `reports/model_metrics.json`
and every PNG below. Headline numbers are from a **held-out 20 % test split**
(222 drugs never seen during training; 708 train / 177 validation / 222 test,
`random_state=42`), so they are honest generalisation numbers, not training fit.

| Metric | Value | What it means |
|---|---|---|
| **Micro-F1 (20 ADR labels, tuned thresholds)** | **0.8095** | Label-weighted detection quality on unseen drugs |
| Micro precision / recall | 0.705 / 0.950 | ~95 % of true ADR labels are caught (recall-first design for safety screening) |
| Macro-F1 | 0.803 | Unweighted mean F1 across the 20 common targets; rare ADRs are not evaluated |
| Mean ROC-AUC | 0.718 | Ranking quality of the raw probabilities per label |
| Mean average precision | 0.804 | Precision holds across the whole ranking |
| Subset accuracy (all 20 labels exact) | 0.131 | Hard multi-label metric, reported for transparency |
| Hamming loss | 0.293 | Fraction of the 20 x 222 prediction cells labelled wrongly |
| **Prevalence baseline micro-F1** | **0.7928** | What "always predict the common ADRs" scores — the model must beat this |
| Alert quality | P@1 0.865 · P@3 0.815 · P@5 0.800 | Mean precision among top 1, 3 and 5 reference-association predictions |

**Model selection** (decided on a separate validation split; the test split was
only ever touched by the winner):

| Candidate | Validation micro-F1 | Train time |
|---|---|---|
| **RandomForest (300 trees) <- served** | **0.8380** | 23.5 s |
| Soft-vote ensemble (RF + ExtraTrees + LogReg) | 0.8351 | 113.8 s |
| Logistic regression | 0.8320 | 5.9 s |
| ExtraTrees | 0.8316 | 28.0 s |

Top per-ADR test results: Nausea F1 0.94 (AUROC 0.85), Headache 0.92, Vomiting 0.89,
Dermatitis 0.89, Rash 0.88 — every one of the eight most prevalent ADRs stays above
F1 0.82 on unseen drugs.

### Generated performance images

These are actual plots from the saved evaluation, not illustrative scores.

![Per-reaction precision, recall, F1 and AUROC](reports/figures/01_per_adr_performance.png)
![Predicted risk-band distribution](reports/figures/02_risk_band_pie.png)
![Model performance dashboard](reports/figures/03_metrics_dashboard.png)
![Distribution of reference ADR associations](reports/figures/04_adr_prevalence_pie.png)
![Threshold sensitivity](reports/figures/05_threshold_tuning.png)
![Validation model comparison](reports/figures/06_model_comparison.png)
![Ranked alert precision and recall](reports/figures/07_alert_quality.png)

**Interpretation:** label-wise accuracy is **70.65%** (1 minus Hamming loss),
whereas exact-match accuracy is **13.06%**. Micro-F1 **80.95% is not accuracy**.
The always-positive baseline is already 79.28% micro-F1; the improvement is only
1.67 percentage points. Common-label imbalance makes F1 alone insufficient.
The pie charts show dataset associations/model outputs, not patient incidence.
Risk-band shares use all 1,107 feature-bearing drugs, including training drugs;
prevalence shows test-set positive associations within the ten displayed labels.
Neither pie is a test-accuracy plot.
The threshold sweep is a test-set diagnostic, not a basis for choosing thresholds.

### Evaluation limitations

- The 20 most common targets and ATC vocabulary were constructed before splitting.
  Although estimator fitting excludes test rows, target selection is not fully
  isolated. These are exploratory internal results, not external validation.
- Random drug splitting may put similar ATC profiles in training and testing.
  Grouped ATC evaluation, repeated splits with confidence intervals and external
  clinical validation are still needed before making generalisation claims.
- All 20 targets are common ADRs. Macro-F1 here says nothing about rare ADR detection.
- SIDER absence is not a confirmed negative. Scores estimate reference-label
  association, not an individual's chance of developing an ADR. No calibration
  or clinical outcome validation has been performed.
- The project fits standard scikit-learn estimators locally; it is **not** using
  a pretrained cloud model. No new neural-network architecture is trained.

## 5. Local setup and operation

From the project directory, in PowerShell (Python 3.11+):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m sentinel.build_model
.\.venv\Scripts\python.exe main.py
```

Open **http://localhost:8000** for the demo UI or **http://localhost:8000/docs**
for interactive API documentation. Installation requires internet; inference,
PDF generation and the baked demo use local data without API keys.
Building all four candidates took about three minutes on the development machine.
The first analysis loads the model lazily; it is not guaranteed to be instant.

The interactive CLI remains available:

```powershell
.\.venv\Scripts\python.exe adr_prediction_model.py
.\.venv\Scripts\python.exe adr_prediction_model.py --drug tigecycline --symptoms nausea,rash --onset-days 3
```

Optional configuration: copy `.env.example` to `.env`. The model artifact is a
trusted local joblib file; never load artifacts supplied by an untrusted party.
To regenerate charts, metrics and the demo, rerun `python -m sentinel.build_model`.
This command refits all candidates and overwrites generated artifacts.

### Docker

```powershell
docker compose up --build -d
docker compose ps
docker compose down
```

The Python 3.11 slim image builds its own model, runs as non-root, exposes port
8000 and checks `/health`. Docker Desktop must be running. Image build needs
internet for packages; a built container can operate offline. Container runtime
verification is not yet confirmed on this machine.

## 6. API and demo workflow

| Endpoint | Purpose |
|---|---|
| `GET /health` | Artifact checks without loading the model |
| `POST /analyze` | Drugs, symptoms and optional onset interval → ADR associations |
| `POST /generate-report` | `drug_result` or enriched `flagged_adr`, `patient`, `output` (`json`/`pdf`) |
| `GET /demo` | Cached Tigecycline analysis with a freshly generated draft |
| `GET /demo/report.pdf` | Download the demo draft without live model inference |
| `GET /model-performance` | Saved evaluation and figure paths |
| `GET /docs` | Interactive request schemas |

Complete PowerShell example, with the server running:

```powershell
$body = @{ drugs = @('tigecycline'); symptoms = @('nausea'); onset_days = 3 } | ConvertTo-Json
$result = Invoke-RestMethod http://localhost:8000/analyze -Method Post -ContentType 'application/json' -Body $body
$report = @{
    drug_result = $result.drugs[0]
    reaction = 'Nausea'
    patient = @{ age = 68; sex = 'Female'; onset_days = 3; therapy_start_date = '2026-09-01' }
    output = 'pdf'
} | ConvertTo-Json -Depth 30
Invoke-WebRequest http://localhost:8000/generate-report -Method Post -ContentType 'application/json' -Body $report -OutFile ICSR_draft.pdf
```

Demo: load the cached case, explain the association scores and synthetic patient,
select **Draft PvPI report**, inspect provenance and missing fields in the PDF,
then show the performance charts. The UI accepts basic patient edits; downloads
require review. It does not submit to PvPI or replace its official reporting portal.

### Clinical and privacy boundaries

- Confirm the generic name. Fuzzy similarity is **not** a verified brand dictionary;
  arbitrary brands are unsupported.
- Naranjo is provisional, not a diagnosis. Unknown answers need review. Never
  rechallenge a patient to improve a score. WHO-UMC requires a separate assessment.
- Keyword severity is a heuristic, not CTCAE grading. Unknown terms currently
  default to mild with an unclassified marker; that is **not** evidence of mildness.
  Seriousness suggestions require independent clinical confirmation.
- Onset interval means time **after starting treatment**, not days before today.
  Without explicit onset or therapy start, the calendar date stays missing.
- Placeholders are not valid reporters. A draft checklist is not proof that
  official minimum reporting criteria are met; observed reactions must be verified.
- Use synthetic data only. Free text can contain PII and is copied into downloads.
  There is no server-side case database. The unauthenticated demo is not suitable
  for public clinical deployment.

## 7. Testing and references

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Tests cover band boundaries, fuzzy rejection, Naranjo weights, severity rules,
provenance, missing dates/reporters, PDFs, integration and API routes. Figures
and metrics are retained for README rendering; cached models and local patient
report downloads are ignored by Git.

Reference starting points (verify current forms and licences before reuse):
[SIDER](http://sideeffects.embl.de/), [WHO ATC/DDD](https://atcddd.fhi.no/),
[Indian Pharmacopoeia Commission / PvPI](https://www.ipc.gov.in/),
[Naranjo et al., 1981](https://doi.org/10.1038/clpt.1981.154).
Background percentages still need specific primary citations.

**Academic prototype only.** Not a certified medical device or a substitute for
professional assessment. Reduced reporting time and improved reporting rates
remain hypotheses, not demonstrated outcomes.

