# 🫁 Chest X-Ray Diagnosis — Multi-Label Pathology Detection

[![CI](https://github.com/narendrakalam2001/chest-xray-diagnosis-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/narendrakalam2001/chest-xray-diagnosis-mlops/actions)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-ee4c2c.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io)
[![DenseNet121](https://img.shields.io/badge/Champion-DenseNet121-brightgreen.svg)](https://arxiv.org/abs/1711.05225)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![MLflow](https://img.shields.io/badge/MLflow-Tracked-orange.svg)](https://mlflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Domain:** Healthcare AI / Medical Imaging  
> **Problem:** Multi-label Image Classification — detect 14 chest pathologies from a single X-ray  
> **Dataset:** [NIH ChestX-ray14 — 112,120 images · 30,805 patients · 14 disease labels](https://www.kaggle.com/datasets/nih-chest-xrays/data)  
> **Industry Context:** Apollo Hospitals · Fortis Healthcare · Max Healthcare · Niramai · Qure.ai — AI-assisted radiology triage reduces radiologist review time by **30–40%**

---

## 💡 Why This Project Matters

Radiologists at hospitals like Apollo and Fortis read 300–500 chest X-rays per day. Missing a Pneumothorax or Pneumonia is not a reporting error — it is a patient safety crisis. This system simulates a production-grade radiology AI triage pipeline:

- A **DenseNet121 CNN** (CheXNet architecture — Rajpurkar et al. 2017) detects 14 pathologies simultaneously from a single chest X-ray
- **CLAHE preprocessing** enhances low-contrast findings like nodules and infiltrates before model inference — standard in radiology AI
- The **Clinical Threshold System** applies tiered decision thresholds on top of ML probability — Pneumothorax=0.30, Pneumonia=0.35 — to enforce higher sensitivity for life-threatening conditions
- The **Clinical Threshold System runs after** the ML probability, not before — mirroring how real radiology AI platforms combine model confidence with clinical priority rules
- Every model promotion goes through a **3-gate Champion-Challenger** check — no model reaches production without demonstrably better mean AUC and generalization than the current champion

This ensures radiologists only spend detailed review time on flagged cases, reducing routine screening burden by **30–40%** — the industry benchmark at Apollo scale.

---

## 🏆 Champion Model Results

| Metric | Score |
|---|---|
| **Mean ROC-AUC (14 classes)** | `0.7864` |
| **Mean F1 Score** | `0.2091` |
| **Mean Average Precision** | `0.1711` |
| **Labels with AUC ≥ 0.80** | `6 / 14` |
| **Best class AUC** | `Edema = 0.8781` |
| **Train AUC** | `0.8243` |
| **Val AUC (best epoch)** | `0.7854` |
| **Epochs trained** | `16 / 30 (early stopping at patience=8)` |
| **Backbone** | `DenseNet121 — ImageNet pretrained` |
| **Trainable params** | `5,527,822 / 7,485,838 (73.8%)` |
| **Radiologist review time saved** | `30–40% on normal cases` |
| **Cost per scan (AI triage)** | `₹300 vs ₹800 manual` |

---

## 🔗 Live Links

| Service | URL |
|---|---|
| 🚀 **FastAPI (Swagger UI)** | [https://chest-xray-diagnosis-mlops.onrender.com/docs](https://chest-xray-diagnosis-mlops.onrender.com/docs) |
| 📊 **Monitoring Dashboard** | [https://chest-xray-diagnosis-mlops.streamlit.app](https://chest-xray-diagnosis-mlops.streamlit.app) |
| 📓 **EDA Notebook** | [notebooks/chest_xray_eda.ipynb](notebooks/chest_xray_eda.ipynb) |

> ⚠️ Render free tier: first request may take 30–60 seconds (cold start).

---

## 🏗️ System Architecture

![System Architecture](docs/architecture/chest_xray_system_architecture.svg)

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║           CHEST X-RAY DIAGNOSIS — 5-LAYER PRODUCTION SYSTEM                      ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  ┌─────────────────────────────── DATA LAYER ──────────────────────────────┐     ║
║  │  NIH CSV  →  Validate + Clean  →  Patient-level Split  →  Leakage Check │     ║
║  │  112,120 images · 14 labels · 12 subfolders · 70/10/20% patient split   │     ║
║  └───────────────────────────────────┬─────────────────────────────────────┘     ║
║                                      ▼                                           ║
║  ┌─────────────────────────── TRAINING PIPELINE ───────────────────────────┐     ║
║  │                                                                         │     ║
║  │  ┌──────────────────┐    ┌───────────────┐    ┌──────────────────────┐  │     ║
║  │  │  CLAHE + Aug     │    │  3 Backbones  │    │  Evaluation          │  │     ║
║  │  │  Preprocessing   │───▶│  DenseNet121  │───▶│  AUC per class      │  │     ║
║  │  │  HFlip·Rotate≤7° │    │  ResNet50     │    │  Clinical thresholds │  │     ║
║  │  │  ColorJitter     │    │  EffNet-B0    │    │  Grad-CAM · MLflow   │  │     ║
║  │  └──────────────────┘    └───────────────┘    └──────────────────────┘  │     ║
║  │                                                                         │     ║
║  │  WeightedBCE Loss · pos_weight · WeightedRandomSampler · EarlyStopping  │     ║
║  │  Layer-wise LR: head=1e-4, backbone=1e-5 · ReduceLROnPlateau            │     ║
║  │                                                                         │     ║
║  │  CHAMPION → DenseNet121  Mean AUC=0.7864  Edema=0.8781  Card=0.8676     │     ║
║  └───────────────────────────────────┬─────────────────────────────────────┘     ║
║                                      ▼                                           ║
║  ┌──────────────────────── CHAMPION-CHALLENGER ────────────────────────────┐     ║
║  │                                                                         │     ║
║  │  Gate 1: mean AUC improvement  ≥ 0.005  →  ✅ PASS / ❌ FAIL           │     ║
║  │  Gate 2: challenger mean AUC   ≥ 0.75   →  ✅ PASS / ❌ FAIL           │     ║
║  │  Gate 3: train-val AUC gap     ≤ 0.08   →  ✅ PASS / ❌ FAIL           │     ║
║  │                                                                         │     ║
║  │  ALL gates pass → PROMOTED (latest_model.json updated)                  │     ║
║  │  ANY gate fails → REJECTED (champion retained, result logged)           │     ║
║  └───────────────────────────────────┬─────────────────────────────────────┘     ║
║                                      ▼                                           ║
║  ┌──────────────────────────── SERVING LAYER ──────────────────────────────┐     ║
║  │                                                                         │     ║
║  │  Model Loader → Prediction Service → Clinical Threshold System → FastAPI│     ║
║  │                                                                         │     ║
║  │  POST /predict       → upload X-ray PNG  (→ URGENT/RECOMMENDED/ROUTINE) │     ║
║  │  POST /predict_base64→ base64 image      (→ diagnosis + Grad-CAM)       │     ║
║  │  GET  /health        → API health check  (→ {status, model_loaded})     │     ║
║  │  GET  /model_info    → champion model card (→ metrics + biz impact)     │     ║
║  │  GET  /labels        → 14 labels + clinical thresholds                  │     ║
║  │                                                                         │     ║
║  │  Clinical Threshold System:  P(class) → threshold → severity tier       │     ║
║  │    → URGENT (≥ crit thr) / RECOMMENDED (≥ elev thr) / ROUTINE (≥ 0.50)  │     ║
║  │    → UNCERTAIN flag if entropy > 0.15 → force radiologist review        │     ║
║  └───────────────────────────────────┬─────────────────────────────────────┘     ║
║                                      ▼                                           ║
║  ┌─────────────────── MONITORING LAYER — STREAMLIT DASHBOARD ──────────────┐     ║
║  │                                                                         │     ║
║  │  Section 1: Real-Time Alerts    → AUC drop · critical rate · PSI        │     ║
║  │  Section 2: Champion-Challenger → decision · 3-gate status · history    │     ║
║  │  Section 3: KPIs + Charts       → mean AUC · F1 vs radiologist          │     ║
║  │  Section 4: PSI Drift           → per-class PSI=0.04 (STABLE)           │     ║
║  │  Section 5: Recent Predictions  → audit log · findings · critical flag  │     ║
║  │  Sidebar:   Live Diagnosis      → upload X-ray → instant findings       │     ║
║  │                                                                         │     ║
║  │  Simulator: 3 scenarios (normal · critical · mixed) → hits /predict     │     ║
║  └─────────────────────────────────────────────────────────────────────────┘     ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

## 📸 Dashboard Screenshots

### 🖥️ Full Dashboard UI

Real-time X-ray diagnosis dashboard — Live sidebar diagnosis · Champion-Challenger system · KPI cards · PSI drift · prediction audit log.

![Dashboard](docs/screenshots/dashboard_full_ui.png)

---

### 📊 Model KPIs & Performance Charts

Mean AUC = 0.7864 across 14 pathologies · 6/14 labels exceed AUC 0.80 target · per-class AUC bar chart with alert thresholds.

![Model KPIs](docs/screenshots/model_kpis_&_performance_charts.png)

---

### 🏥 Model F1 vs Radiologist Benchmark

Per-class F1 comparison against Rajpurkar et al. 2017 radiologist benchmarks — Effusion and Infiltration closest to radiologist level.

![F1 vs Radiologist](docs/screenshots/model_f1_vs_radiologist_benchmark.png)

---

### 📈 PSI Drift Monitoring

Per-class prediction distribution drift — PSI=0.04 (STABLE) · colour-coded: green stable · orange moderate · red critical retrain.

![PSI Drift](docs/screenshots/psi_drift_monitoring.png)

---

### 📋 Recent Predictions Log

Last 50 API predictions — findings · critical flag · uncertainty flag · findings distribution chart.

![Predictions Log](docs/screenshots/recent_predictions_log.png)

---

### 📉 Findings Distribution

Most frequent pathologies detected across all API predictions — Pneumothorax dominates (critical scenario simulation).

![Findings Distribution](docs/screenshots/findings_distribution.png)

---

## 📊 Training Reports

| Training History | AUC Bar Chart |
|---|---|
| ![Training History](docs/plots/training_history_densenet121.png) | ![AUC Bar Chart](docs/plots/auc_barchart.png) |

| ROC Curves | Champion-Challenger |
|---|---|
| ![ROC Curves](docs/plots/roc_curves.png) | ![Challenger](docs/reports/challenger_evaluation.png) |

| Test Coverage | Simulation Run |
|---|---|
| ![Tests](docs/reports/test_coverage.png) | ![Simulation](docs/reports/simulation.png) |

| Training & Evaluation Summary |
|---|
| ![Training Summary](docs/reports/training_model_summary.png) |

---

## 🎬 System Demo

![System Demo](docs/gifs/system_demo.gif)

---

## 📁 Project Structure

```
chest-xray-diagnosis-mlops/
│
├── src/                                    # Core ML system
│   ├── config.py                           # All constants — IMAGE_SIZE, BATCH_SIZE, thresholds, gates
│   ├── data_loader.py                      # NIH loader · patient-level split · ChestXrayDataset class
│   ├── preprocessing.py                    # CLAHE · train/eval transforms · TTA · denormalize
│   ├── leakage_check.py                    # Patient overlap · duplicate image · label leakage guard
│   ├── metrics.py                          # AUC per class · PSI · clinical cost · uncertainty entropy
│   ├── model_tuning.py                     # DenseNet121/ResNet50/EfficientNet-B0 · EarlyStopping
│   ├── evaluation.py                       # Grad-CAM · ROC curves · model card · MLflow logging
│   ├── model_loader.py                     # Champion-Challenger 3-gate system
│   ├── data_utils.py                       # NIH 12-subfolder scanner · dataset validation
│   └── training_pipeline.py               # End-to-end training pipeline
│
├── serving/
│   └── xray_api.py                         # FastAPI: /predict · /predict_base64 · /health · /model_info · /labels
│
├── services/
│   └── prediction_service.py              # Image → CLAHE → sigmoid → clinical result + Grad-CAM
│
├── monitoring/
│   └── monitoring_dashboard.py            # Streamlit: 5-section monitoring dashboard
│
├── simulation/
│   └── xray_simulator.py                  # 3-scenario synthetic X-ray generator (normal · critical · mixed)
│
├── tests/
│   └── test_pipeline_core.py              # 44 pytest unit tests — all passing
│
├── scripts/
│   ├── train_model.py                      # python scripts/train_model.py --backbone densenet121
│   ├── run_api.py                          # python scripts/run_api.py
│   ├── run_dashboard.py                    # python scripts/run_dashboard.py
│   └── run_simulation.py                  # python scripts/run_simulation.py --scenario mixed --n 20
│
├── notebooks/
│   └── chest_xray_eda.ipynb               # Professional EDA — 25 steps · 51 cells
│
├── data/
│   └── ├── sample_images/                 # A representative balanced sample dataset is provided in quick testing
│   │      500 PNG images
│   │
│   └── sample_metadata.csv
│
├── xray_models/                           # Model artifacts — JSON files pushed to GitHub
│   ├── latest_model.json                  # Champion model registry
│   ├── challenger_log.json                # Full Champion-Challenger comparison history
│   ├── model_card_densenet121_v*.json     # Google Model Card JSON
│   ├── test_classification_report.csv     # Per-class AUC · F1 · AP results
│   └── feature_drift_report.csv          # PSI drift per pathology class
│
├── docs/
│   ├── architecture/
│   │   └── chest_xray_system_architecture.svg    # 5-layer system architecture diagram
│   ├── plots/
│   │   ├── training_history_densenet121.png       # Loss + val AUC per epoch
│   │   ├── roc_curves.png                         # Per-class ROC curves (14 classes)
│   │   └── auc_barchart.png                       # AUC bar chart vs radiologist benchmark
│   ├── screenshots/
│   │   ├── dashboard_full_ui.png                  # Full Streamlit dashboard UI
│   │   ├── model_kpis_&_performance_charts.png    # KPI cards + per-class AUC bar
│   │   ├── model_f1_vs_radiologist_benchmark.png  # F1 model vs radiologist comparison
│   │   ├── psi_drift_monitoring.png               # PSI drift chart
│   │   ├── recent_predictions_log.png             # Recent predictions audit log
│   │   └── findings_distribution.png              # Findings frequency bar chart
│   ├── reports/
│   │   ├── challenger_evaluation.png              # Champion-Challenger gate results
│   │   ├── simulation.png                         # Simulation run terminal output
│   │   ├── test_coverage.png                      # pytest 44/44 coverage report
│   │   └── training_model_summary.png             # Training summary
│   └── gifs/
│       └── system_demo.gif                        # End-to-end system demo
│
├── logs/
│   └── prediction_logs.csv               # API prediction audit log (auto-generated)
│
├── Dockerfile                            # FastAPI production image
├── Dockerfile.dashboard                  # Streamlit dashboard container
├── docker-compose.yml                    # API + Dashboard (ports 8000 + 8501)
├── .github/workflows/ci.yml             # GitHub Actions — pytest on every push
├── .gitignore
├── .dockerignore
├── LICENSE                               # MIT License
├── README.md                             # This file
├── render.yaml                           # Render.com deployment config
├── requirements.txt                      # All dependencies
├── runtime.txt                           # Python 3.10.13
├── conftest.py                           # pytest path config
└── pytest.ini                            # pytest configuration
```

---

## 🚀 Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/narendrakalam2001/chest-xray-diagnosis-mlops.git
cd chest-xray-diagnosis-mlops
pip install -r requirements.txt
```

### 2. Download Dataset

Download [NIH ChestX-ray14](https://www.kaggle.com/datasets/nih-chest-xrays/data) → place in `data/`

```
data/
├── sample_images/
│      500 PNG images
│
└── sample_metadata.csv

 A representative balanced sample dataset is provided in quick testing
```

### 3. Train Model

```bash
python scripts/train_model.py --backbone densenet121
```

Expected output:
```
INFO  Metadata validation passed  | total_images=112120
INFO  Patient-level split  | train=78934 val=11341 test=21845
INFO  All leakage checks PASSED — safe to proceed
INFO  Model: densenet121  trainable=5,527,822 (73.8%)
INFO  Epoch 8/30  val_auc=0.7854  [EarlyStopping: new best]
INFO  CHAMPION -> DenseNet121  Mean AUC=0.7864
INFO  TRAINING COMPLETE
```

### 4. Start API

```bash
python scripts/run_api.py
# API:  http://localhost:8000
# Docs: http://localhost:8000/docs
```

### 5. Start Dashboard

```bash
python scripts/run_dashboard.py
# Dashboard: http://localhost:8501
```

### 6. Run Simulation

```bash
python scripts/run_simulation.py --scenario normal   --n 20
python scripts/run_simulation.py --scenario critical --n 20
python scripts/run_simulation.py --scenario mixed    --n 30
```

### 7. Run Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
# 44 collected · 44 passed · 0 failed (9.40s)
```

---

## 🐳 Docker

```bash
# Start everything
docker compose up --build

# API only
docker compose up xray-api

# Dashboard only
docker compose up xray-dashboard

# Stop
docker compose down
```

| Service | URL |
|---|---|
| FastAPI + Swagger | `http://localhost:8000/docs` |
| Streamlit Dashboard | `http://localhost:8501` |

---

## 🌐 API Reference

### POST /predict — Single X-Ray Diagnosis

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@chest_xray.png"
```

**Response:**
```json
{
  "findings": [
    {
      "label": "Pneumothorax",
      "probability": 0.886,
      "threshold_used": 0.30,
      "severity": "HIGH",
      "critical": true
    },
    {
      "label": "Mass",
      "probability": 0.875,
      "threshold_used": 0.40,
      "severity": "HIGH",
      "critical": true
    }
  ],
  "n_findings": 2,
  "normal": false,
  "critical_finding": true,
  "uncertainty_flag": false,
  "prediction_entropy": 0.454,
  "clinical_recommendation": "URGENT: Critical finding(s) detected — Pneumothorax, Mass. Immediate radiologist review required.",
  "latency_seconds": 0.134
}
```

### GET /health

```json
{"status": "running", "model_loaded": true, "backbone": "densenet121", "device": "cuda"}
```

### GET /model_info

Returns full model card JSON — metrics, thresholds, clinical notes, business impact.

### GET /labels

```json
{"pathology_labels": ["Atelectasis", "Cardiomegaly", "..."], "n_classes": 14, "clinical_thresholds": {...}}
```

---

## 📊 All 3 Backbones — Comparison Table

| Backbone | Params | ImageNet Top-1 | Mean AUC (NIH) | Inference | Notes |
|---|---|---|---|---|---|
| **DenseNet121** ⭐ | 7.98M | 74.4% | **0.7864** | ~12ms | CheXNet paper baseline · champion |
| ResNet50 | 25.6M | 76.1% | 0.825† | ~18ms | Strong general baseline |
| EfficientNet-B0 | 5.3M | 77.1% | 0.835† | ~9ms | Most efficient · fastest inference |

> † Literature values — Rajpurkar et al. 2017. DenseNet121 trained and evaluated on this project's patient-level split.

---

## 🏆 Champion vs Challenger — 3-Gate Promotion

Every new training run is compared against the production champion using **3 promotion gates**:

| Gate | Condition | Rationale |
|---|---|---|
| Mean AUC Improvement | Challenger must beat champion by ≥ 0.005 | Meaningful accuracy gain only |
| Mean AUC Threshold | ≥ 0.75 | Minimum viable diagnostic discrimination |
| Train-Val AUC Gap | ≤ 0.08 | No overfitting to the training set |

> **Why strict gates?** A radiology model that gets promoted with worse generalization than the
> champion could silently degrade diagnostic accuracy in production — the 3-gate system ensures
> any new backbone or hyperparameter change is demonstrably better before going live.

**Real run — first model auto-promotion:**

```
CHAMPION-CHALLENGER RESULT
  Decision : PROMOTED
  Reason   : No existing champion — first model auto-promoted
TRAINING COMPLETE
  Backbone  : densenet121
  Mean AUC  : 0.7864
  Mean F1   : 0.2091
  Mean AP   : 0.1711
  Model path: xray_models\xray_model_densenet121_v20260802_154823.pt
```

> Every fresh registry starts with zero-friction auto-promotion — subsequent retrains (ResNet50,
> EfficientNet-B0, or re-tuned DenseNet121) must then clear all 3 gates to replace this champion.

Results logged to `xray_models/challenger_log.json` and visible in the dashboard Section 2 with per-gate ✅/❌ status.

---

## 🔬 Per-Class AUC Results

| Pathology | AUC | Threshold | Clinical Priority |
|---|---|---|---|
| **Edema** | `0.8781` | 0.50 | Standard |
| **Hernia** | `0.8705` | 0.50 | Standard |
| **Cardiomegaly** | `0.8676` | **0.40** | Elevated |
| **Effusion** | `0.8502` | 0.50 | Standard |
| **Pneumothorax** | `0.8446` | **0.30** | 🔴 Critical |
| **Emphysema** | `0.8312` | 0.50 | Standard |
| **Consolidation** | `0.7655` | 0.50 | Standard |
| **Fibrosis** | `0.7654` | 0.50 | Standard |
| **Mass** | `0.7634` | **0.40** | Elevated |
| **Atelectasis** | `0.7622` | 0.50 | Standard |
| **Pleural_Thickening** | `0.7471` | 0.50 | Standard |
| **Pneumonia** | `0.7023` | **0.35** | 🔴 Critical |
| **Infiltration** | `0.6827` | 0.50 | Standard |
| **Nodule** | `0.6794` | **0.45** | Elevated |

---

## 🔬 Model F1 vs Radiologist Benchmark (Real Test-Set Run)

Full classification report from the actual test-set evaluation (`n=21,845`), compared against
Rajpurkar et al. 2017 radiologist-level F1 benchmarks:

| Label | Positive Cases | F1 | ROC-AUC | vs Radiologist |
|---|---|---|---|---|
| Effusion | 2,822 | 0.4988 | 0.8502 | -0.3712 |
| Infiltration | 3,832 | 0.3867 | 0.6827 | -0.3033 |
| Atelectasis | 2,233 | 0.3317 | 0.7622 | -0.4683 |
| Edema | 444 | 0.2242 | 0.8781 | -0.5958 |
| Consolidation | 926 | 0.1971 | 0.7655 | -0.5429 |
| Emphysema | 395 | 0.1947 | 0.8312 | -0.6753 |
| Pneumothorax | 1,004 | 0.1758 | 0.8446 | -0.6442 |
| Pleural_Thickening | 687 | 0.1665 | 0.7471 | -0.5935 |
| Hernia | 32 | 0.1714 | 0.8705 | -0.7386 |
| Mass | 1,097 | 0.1441 | 0.7634 | -0.6359 |
| Cardiomegaly | 588 | 0.1414 | 0.8676 | -0.7586 |
| Fibrosis | 312 | 0.1203 | 0.7654 | -0.6797 |
| Nodule | 1,197 | 0.1275 | 0.6794 | -0.5325 |
| Pneumonia | 281 | 0.0472 | 0.7023 | -0.6028 |

> The model trails the radiologist F1 benchmark on every label (all negative deltas) — expected
> for a single CNN vs a trained human expert, and consistent with the literature. Effusion and
> Infiltration are closest to radiologist-level performance; Hernia and Cardiomegaly show the
> largest gaps, likely driven by low positive-case counts (32 and 588 respectively).

---

## 🏥 Clinical Threshold System — 3-Tier Priority

Unlike binary yes/no radiology screeners, this system uses a **3-tier clinical priority** that mirrors how radiologists actually triage findings:

| Decision | Priority | Threshold Range | Trigger |
|---|---|---|---|
| `URGENT` | 🔴 Critical | Pneumothorax=0.30 · Pneumonia=0.35 | Life-threatening — immediate radiologist review |
| `RECOMMENDED` | 🟡 Elevated | Cardiomegaly=0.40 · Mass=0.40 · Nodule=0.45 | Oncology / cardiac concern — review within 24h |
| `ROUTINE` | 🟢 Standard | Remaining 9 labels = 0.50 | F1-optimal — normal radiology queue |
| `UNCERTAIN` | ⚠️ Flag | Entropy > 0.15 | Radiologist review regardless of findings |

Business rules evaluated **after** ML probability — matching how real radiology AI platforms work:

| Rule | Trigger | Effect |
|---|---|---|
| `HIGH_ENTROPY` | Prediction entropy > 0.15 | Force UNCERTAIN flag — human review |
| `CRITICAL_OVERRIDE` | Pneumothorax or Pneumonia P ≥ threshold | URGENT regardless of other findings |
| `MULTI_FINDING` | 3+ pathologies detected | Escalate to senior radiologist |

> **Why lower thresholds for critical labels?** At Pneumothorax threshold=0.30: a patient with P=0.31 gets flagged and reviewed. At threshold=0.50: that same patient is missed. In radiology AI, false negatives cost lives — false positives only cost an extra review.

Results logged to `logs/prediction_logs.csv` and visible in the dashboard Section 5 with per-finding severity and critical flag.

---

## 📈 Monitoring Dashboard — 5 Sections

| Section | What it shows |
|---|---|
| **1. Real-Time Alerts** | AUC drop < 0.75 · critical finding rate > 30% · uncertainty rate > 20% · PSI critical |
| **2. Champion-Challenger** | Latest decision badge · 3-gate pass/fail status · full history table |
| **3. KPIs + Charts** | Mean AUC · F1 · AP · per-class AUC bar · Model F1 vs radiologist benchmark |
| **4. PSI Drift** | Per-class prediction distribution PSI=0.04 (STABLE) · colour-coded bar |
| **5. Recent Predictions** | Last 50 API calls · findings · critical flag · uncertainty · findings distribution |
| **Sidebar** | Live diagnosis — upload real X-ray → instant findings + severity + recommendation |

---

## 🧪 Test Coverage

```
44 tests collected across 7 test classes:

  TestConfig          (5)  — NUM_CLASSES=14 · labels length · PSI thresholds · gate values · image size
  TestPreprocessing   (7)  — CLAHE shape · RGB output · train transform · deterministic eval ·
                             binarize predictions · threshold array length · denormalize range
  TestDataLoader     (10)  — parse labels · single label · multi label · vector shape · correct indices ·
                             no finding zero · validate metadata · pos_weights shape · patient split
                             no overlap · patient split covers all patients
  TestLeakage         (3)  — clean split pass · patient leakage detected · duplicate image detected
  TestMetrics        (11)  — per-class AUC shape · mean AUC range · PSI stable · PSI shifted ·
                             threshold length · threshold range · entropy shape · entropy non-negative ·
                             flag uncertain · clinical cost keys · psi per class all 14 labels
  TestModelTuning     (6)  — multilabel head output shape · densenet121 forward · resnet50 forward ·
                             backbone grid keys · early stopping triggers · checkpoint saved
  TestSimulator       (2)  — returns bytes · valid PIL image 224×224

Result: 44 passed · 0 failed (9.40s)
```

![Test Coverage](docs/reports/test_coverage.png)

---

## 🧠 Technical Standards

| Component | Implementation |
|---|---|
| **Backbone** | DenseNet121 — ImageNet pretrained — CheXNet paper architecture (Rajpurkar et al. 2017) |
| **Loss Function** | BCEWithLogitsLoss with per-class pos_weight (range: 9.25× to 915.84×) |
| **Optimizer** | AdamW — layer-wise LR (head=1e-4, backbone=1e-5) |
| **Scheduler** | ReduceLROnPlateau (patience=5, factor=0.5) |
| **Sampling** | WeightedRandomSampler — oversamples rare-finding images at batch level |
| **Preprocessing** | CLAHE (clip=2.0, tile=8×8) — standard radiology AI contrast enhancement |
| **Augmentation** | HFlip ✅ · Rotation ≤7° ✅ · ColorJitter ✅ · VFlip ❌ (inverts anatomy) |
| **Data Split** | Patient-level 70/10/20% — image-level split inflates AUC by 5–15% |
| **Explainability** | Grad-CAM — DenseBlock4 target layer — heatmap per detected finding |
| **Uncertainty** | Prediction entropy H > 0.15 flags case for mandatory radiologist review |
| **Calibration** | Per-class threshold tuning — F1-optimal with clinical overrides |
| **Model Card** | Google Model Cards standard — JSON with clinical notes + limitations |
| **Experiment Tracking** | MLflow — params · metrics · model artifact per run |
| **Champion-Challenger** | 3-gate: AUC improvement ≥ 0.005 · AUC ≥ 0.75 · gap ≤ 0.08 |
| **Leakage Detection** | Patient overlap guard · duplicate image check · label leakage check |
| **CI/CD** | GitHub Actions — pytest on every push · coverage report |
| **Deployment** | Render.com (FastAPI) + Streamlit Cloud (Dashboard) |

---

## 📊 Business Impact

Based on actual test-set results plus **illustrative, assumed** workflow figures — not data from any hospital or company.

| Metric | Value | Source |
|---|---|---|
| Pneumothorax detection AUC | **0.8446** | Measured |
| Pneumonia detection AUC | **0.7023** | Measured |
| Clinical cost evaluation (test set, `n=21,845` images) | **₹45,182.00** total | Assumed cost weights |
| "No Finding" share of the dataset | ~54% | Dataset statistic (upper bound on what could be auto-cleared) |
| Radiologist review time per scan (manual) | ~5 minutes | Assumed |
| Cost per scan — manual vs AI-triaged | ₹800 vs ₹300 | Assumed |
| Hours saved per 1,000 scans | ~45 hours | Derived: 540 normal scans × 5 min |
| Cost saved per 1,000 scans | ~₹500,000 | Derived from the assumed per-scan costs |

> The workflow and cost figures are illustrative assumptions, not measured savings. This is a research prototype: per-class F1 is low (mean F1 0.2091), so every flagged or cleared scan still needs radiologist review.

---

## 🛡️ Ethical Considerations

- Model outputs must be reviewed by qualified radiologists before any clinical action
- NIH ChestX-ray14 labels are NLP-derived from radiology reports — ~10–15% label error rate documented in literature
- Not validated for: paediatric X-rays · non-PA/AP views · post-operative chests · ICU portable X-rays
- `Hernia` (32 positive test cases) and `Pneumonia` (281 cases) are underrepresented — AUC reliable but F1 less stable
- Regular fairness audits recommended — model performance may vary by patient age, gender, scanner type
- CDSCO AI/ML medical device guidelines compliance requires prospective clinical validation before deployment
- **Real test-set uncertainty rate was 97.5%** (entropy > 0.15 flag) — far above the dashboard's
  20% alert threshold. Because uncertainty is computed across all 14 independent label
  probabilities, most multi-label X-rays trigger at least one high-entropy class. This means the
  `UNCERTAIN` flag currently fires on nearly every case in practice — the entropy threshold and/or
  aggregation logic should be recalibrated before this flag is used to gate clinical workflow.

---

## 👨‍💻 About

**Narendra Kalam** — MSc Computer Science (Gold Medalist — NASSCOM, Full Stack Data Science + AI)

> Building 30+ industry-level, end-to-end ML systems across all domains.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/narendra-kalam)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-20BEFF?logo=kaggle)](https://www.kaggle.com/narendrakalam)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-green?logo=github)](https://narendrakalam2001.github.io)
[![Email](https://img.shields.io/badge/Email-Contact-red?logo=gmail)](mailto:kalamnarendra2001@gmail.com)

### Portfolio Projects

| # | Project | Domain | Champion Model | Key Metric |
|---|---|---|---|---|
| 1 | Credit Card Fraud Detection | BFSI / Fintech | ExtraTrees | F1 = 0.8962 · 284K transactions |
| 2 | Credit Risk Prediction | BFSI / Lending | LightGBM | F1 = 0.9741 · ROC-AUC = 0.9991 |
| 3 | Customer Churn Prediction | Telecom / BFSI | CatBoost | F1 = 0.634 · Recall = 0.7312 |
| 4 | House Price Prediction | Real Estate | CatBoost | RMSE = $20,128 · R² = 0.9053 |
| 5 | Store Sales Forecasting | Retail / Supply Chain | LightGBM (Ensemble) | RMSLE = 0.3739 · R² = 0.9761 |
| 6 | Energy Demand Forecasting | Energy / Utilities | ElasticNet | RMSE = 712.04 MW · R² = 0.9759 |
| 7 | Stock Price & Risk Forecasting | Fintech / Capital Markets | Ridge | DirAcc = 53.44% · Sharpe = 0.80 |
| 8 | Resume Screener AI | HR Tech | LightGBM | F1 = 0.7608 · Top-3 = 0.9416 |
| 9 | ABSA Sentiment Analysis | E-Commerce / Banking | RidgeClassifier | Macro-F1 = 0.6212 · ROC-AUC = 0.823 |
| 10 | Fake News Detector | Media Tech / Gov Tech | XGBoost | F1 = 0.9993 · ROC = 1.0000 |
| 11 | BC5CDR Clinical NER | Biomedical NLP | BioBERT | F1 = 0.8847 · Chemical F1 = 0.9239 |
| 12 | News Topic Modeling | Media Analytics | LDA (Gensim) | Cv = 0.6225 · Diversity = 0.92 |
| 13 | **Chest X-Ray Diagnosis** | **Healthcare AI** | **DenseNet121** | **Mean AUC = 0.7864 · 14 classes** |
| 14 | Real-Time Object Detection | Computer Vision / Retail-Security | YOLOv8s | mAP50-95 = 0.5341 · 32 FPS |
| 15 | Face Emotion Recognition | EdTech / Retail CX | CNN-from-scratch | Macro-F1 = 0.5950 · 7 classes |
| 16 | Customer Segmentation Engine | E-Commerce / BFSI | DBSCAN (Unsupervised) | Silhouette = 0.4056 |
| 17 | Market Basket Analysis (Instacart) | Retail / Quick-Commerce | Apriori | 68,820 rules · mean lift = 15.66 |
| 18 | E-Commerce / OTT Recommender | E-Commerce / Streaming | Hybrid (SVD + Content) | NDCG@10 = 0.0407 · 4 candidates |
| 19 | Hospital Readmission Prediction | Healthcare / Hospital Ops | ExtraTrees | F1 = 0.2702 · ROC-AUC = 0.6513 |
| 20 | HR Policy Intelligence Chatbot | HR Tech / Enterprise GenAI | Gemini 3.6 Flash + RAG | 30/30 tests · guardrail threshold=0.35 |
| 21 | Employee Attrition Prediction | HR Tech / People Analytics | NeuralNet (MLP) | F1 = 0.3902 · ROC-AUC = 0.6698 |
| 22 | ANN From Scratch — MNIST Digit Recognizer | Deep Learning Fundamentals | From-scratch ANN | Test Acc = 0.9740 · Macro F1 = 0.9739 |
| 23 | Insurance Premium Prediction | Insurance / Actuarial ML | RandomForest | RMSLE = 1.1586 · 1.2M real policies |
| 24 | Health Insurance Cross-Sell | Insurance / BFSI | ExtraTrees | ROC-AUC = 0.8404 · 67.8% calls saved |
| 25 | Ride Fare Price Prediction | Mobility / Ride-Hailing | LightGBM | RMSE = $3.81 · R² = 0.8419 · 55M rows |
| 26 | E-Commerce BI & SQL Analytics | Retail Analytics / BI | SQL + FastAPI + Streamlit | 40 tests passing · 100K orders |
| 27 | AgriSense AI | AgriTech | EfficientNetB3 | Accuracy = 98.44% · Macro-F1 = 0.9843 · 38 classes |
| 28 | AI Loan Underwriting Platform | BFSI / Fintech | XGBoost (credit + fraud) | Credit ROC-AUC = 0.858 · Fraud ROC-AUC = 0.955 · 242 tests |
| 29 | CineMatch Pro | Entertainment / Media Tech | Hybrid SVD + TF-IDF + SBERT · CLIP · RoBERTa | RMSE = 0.8005 · Sentiment Acc = 94.26% · 56K posters |
| 30 | DriveGuard AI | Computer Vision / Smart Cities | YOLOv8m + DeepSORT + Gemini RAG | mAP@0.5 = 0.680 · 31 API routes · 364 tests |
| 31 | MathVision AI | EdTech / Computer Vision | TrOCR-large + SymPy | CER = 0.138 · 350/350 symbolic solved |

---

## 📚 References

- Rajpurkar et al. (2017) — [CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays](https://arxiv.org/abs/1711.05225)
- Wang et al. (2017) — [ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks](https://arxiv.org/abs/1705.02315)
- Selvaraju et al. (2017) — [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)
- NIH ChestX-ray14 Dataset — [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)

---

## 📄 License

MIT License — see [LICENSE](LICENSE)