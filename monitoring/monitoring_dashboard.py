# ============================================================
# MONITORING DASHBOARD — Chest X-Ray Diagnosis ML System
# ============================================================
# 5 Sections:
#   1. Real-Time Alerts (AUC drop, critical finding rate, drift)
#   2. Champion vs Challenger History
#   3. KPIs + Charts (mean AUC, findings distribution, per-class AUC)
#   4. PSI Drift Monitoring (per-class prediction drift)
#   5. Recent Predictions Log
# ============================================================

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import base64
import io
from PIL import Image

st.set_page_config(page_title="Chest X-Ray Diagnosis Dashboard", layout="wide")
st.title("🫁 Chest X-Ray Diagnosis — Monitoring Dashboard")

# ── API URL ───────────────────────────────────────────────────
API_URL = os.getenv(
    "XRAY_API_URL",
    "https://chest-xray-diagnosis-api.onrender.com"
)

PATHOLOGY_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

# ── PSI thresholds ────────────────────────────────────────────
PSI_MODERATE  = 0.10
PSI_HIGH      = 0.20
MEAN_AUC_ALERT = 0.75

# ── Path resolution ───────────────────────────────────────────
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR    = os.path.dirname(_SCRIPT_DIR)
except Exception:
    BASE_DIR = os.getcwd()

LOG_PATH        = os.path.join(BASE_DIR, "logs",        "prediction_logs.csv")
PSI_PATH        = os.path.join(BASE_DIR, "xray_models", "feature_drift_report.csv")
CHALLENGER_PATH = os.path.join(BASE_DIR, "xray_models", "challenger_log.json")
METRICS_PATH    = os.path.join(BASE_DIR, "xray_models", "test_classification_report.csv")
MODEL_PATH      = os.path.join(BASE_DIR, "xray_models", "latest_model.json")


# ============================================================
# SIDEBAR — LIVE PREDICTION
# ============================================================

st.sidebar.header("🔬 Live X-Ray Diagnosis")
st.sidebar.markdown("Upload a chest X-ray image to get AI diagnosis.")

uploaded_file = st.sidebar.file_uploader(
    "Upload Chest X-Ray (PNG/JPG)",
    type=["png", "jpg", "jpeg"],
    key="xray_upload"
)

if uploaded_file and st.sidebar.button("Run Diagnosis"):
    with st.sidebar:
        with st.spinner("Running AI diagnosis... (First request may take 30-60s — Render cold start)"):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                    timeout=90,
                )
                if response.status_code == 200:
                    result = response.json()
                    findings = result.get("findings", [])

                    st.success("Diagnosis complete!")

                    if result.get("normal"):
                        st.info("✅ No pathology detected")
                    else:
                        st.warning(f"⚠️ {len(findings)} finding(s) detected")

                    if result.get("critical_finding"):
                        st.error("🔴 CRITICAL FINDING — Immediate radiologist review")

                    for f in findings:
                        color = "red" if f.get("critical") else "orange"
                        st.markdown(
                            f"<span style='color:{color}'>**{f['label']}**</span>: "
                            f"P={f['probability']:.3f}  [{f['severity']}]",
                            unsafe_allow_html=True
                        )

                    st.caption(result.get("clinical_recommendation", ""))

                    if result.get("uncertainty_flag"):
                        st.warning("⚠️ High uncertainty — radiologist review recommended")
                else:
                    st.error(f"API error: HTTP {response.status_code}")
                    st.code(response.text[:300])
            except requests.exceptions.Timeout:
                st.warning("Request timed out (90s). Render is waking up. Try again in 30s.")
            except Exception as e:
                st.error(f"Connection error: {e}")


# ============================================================
# SECTION 1 — REAL-TIME ALERTS
# ============================================================

st.markdown("---")
st.subheader("🚨 Real-Time Monitoring Alerts")

alerts_found = False

# ── Log-based alerts ──────────────────────────────────────────
if os.path.exists(LOG_PATH):
    try:
        df_log = pd.read_csv(LOG_PATH)

        if "critical_finding" in df_log.columns:
            critical_rate = df_log["critical_finding"].mean()
            if critical_rate > 0.30:
                st.error(f"🔴 HIGH CRITICAL FINDING RATE: {critical_rate:.1%} (expected < 30%)")
                alerts_found = True

        if "uncertainty_flag" in df_log.columns:
            uncertain_rate = df_log["uncertainty_flag"].mean()
            if uncertain_rate > 0.20:
                st.warning(f"🟡 HIGH UNCERTAINTY RATE: {uncertain_rate:.1%}. Consider retraining.")
                alerts_found = True

        if "n_findings" in df_log.columns:
            avg_findings = df_log["n_findings"].mean()
            if avg_findings > 5:
                st.warning(f"🟡 HIGH AVG FINDINGS: {avg_findings:.1f} per scan (possible model drift)")
                alerts_found = True

    except Exception as e:
        st.warning(f"Could not load prediction log: {e}")

# ── PSI drift alerts ──────────────────────────────────────────
if os.path.exists(PSI_PATH):
    try:
        df_psi = pd.read_csv(PSI_PATH)
        if "drift_score" in df_psi.columns and "feature" in df_psi.columns:
            max_psi     = df_psi["drift_score"].max()
            top_feature = df_psi.iloc[df_psi["drift_score"].idxmax()]["feature"]
            if max_psi >= PSI_HIGH:
                st.error(f"🔴 CRITICAL DRIFT: PSI={max_psi:.4f} on '{top_feature}'. Retrain model.")
                alerts_found = True
            elif max_psi >= PSI_MODERATE:
                st.warning(f"🟡 MODERATE DRIFT: PSI={max_psi:.4f} on '{top_feature}'. Monitor closely.")
                alerts_found = True
    except Exception:
        pass

# ── Model card AUC alert ──────────────────────────────────────
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH) as f:
            reg = json.load(f)
        card_path = reg.get("model_card_path", "")
        if card_path and os.path.exists(card_path):
            with open(card_path) as f:
                card = json.load(f)
            mean_auc = card.get("metrics", {}).get("mean_auc", 1.0)
            if mean_auc < MEAN_AUC_ALERT:
                st.error(f"🔴 LOW MODEL AUC: {mean_auc:.4f} < {MEAN_AUC_ALERT}. Retrain recommended.")
                alerts_found = True
    except Exception:
        pass

if not alerts_found:
    st.success("✅ All systems normal — no alerts triggered")


# ============================================================
# SECTION 2 — CHAMPION vs CHALLENGER HISTORY
# ============================================================

st.markdown("---")
st.subheader("🏆 Champion vs Challenger History")

if os.path.exists(CHALLENGER_PATH):
    try:
        with open(CHALLENGER_PATH) as f:
            challenger_log = json.load(f)

        if challenger_log:
            latest         = challenger_log[-1]
            decision_color = "green" if latest["decision"] == "PROMOTED" else "red"
            icon           = "✅" if latest["decision"] == "PROMOTED" else "❌"

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Latest Challenger",    latest.get("challenger_name", "—"))
            with col2:
                st.metric("Challenger Mean AUC",  latest.get("challenger_mean_auc", "—"))
            with col3:
                st.metric("Champion Mean AUC",    latest.get("champion_mean_auc",   "—") or "First Run")
            with col4:
                st.markdown(f"**{icon} Decision:** <span style='color:{decision_color}'>"
                            f"{latest['decision']}</span>", unsafe_allow_html=True)

            # Show full gate breakdown
            gates = latest.get("gates", {})
            if gates:
                st.markdown("**Gate Results:**")
                gate_items = {
                    "AUC Improvement":  gates.get("auc_improvement_passed", False),
                    "AUC Threshold":    gates.get("auc_threshold_passed",   False),
                    "Generalization Gap": gates.get("gap_passed",          False),
                }
                for gate_name, passed in gate_items.items():
                    icon_g = "✅" if passed else "❌"
                    st.markdown(f"  {icon_g} {gate_name}")

            # History table
            if len(challenger_log) > 1:
                st.markdown("**Full History:**")
                hist_df = pd.DataFrame(challenger_log)[
                    ["evaluated_at", "challenger_name", "challenger_mean_auc",
                     "champion_mean_auc", "decision"]
                ]
                st.dataframe(hist_df, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not load challenger log: {e}")
else:
    st.info("No challenger log found. Train a model first.")


# ============================================================
# SECTION 3 — KPIs + CHARTS
# ============================================================

st.markdown("---")
st.subheader("📊 Model KPIs & Performance Charts")

if os.path.exists(METRICS_PATH):
    try:
        metrics_df = pd.read_csv(METRICS_PATH)

        # ── KPI row ──────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            mean_auc = metrics_df["roc_auc"].dropna().mean()
            st.metric("Mean AUC (14 classes)", f"{mean_auc:.4f}")
        with col2:
            mean_f1 = metrics_df["f1"].dropna().mean()
            st.metric("Mean F1 Score", f"{mean_f1:.4f}")
        with col3:
            mean_ap = metrics_df["avg_precision"].dropna().mean() if "avg_precision" in metrics_df.columns else 0
            st.metric("Mean Avg Precision", f"{mean_ap:.4f}")
        with col4:
            n_above_08 = (metrics_df["roc_auc"].dropna() >= 0.80).sum()
            st.metric("Labels with AUC ≥ 0.80", f"{n_above_08}/14")

        # ── Per-class AUC bar chart ───────────────────────────
        st.markdown("**Per-Class AUC:**")
        fig, ax = plt.subplots(figsize=(14, 5))
        auc_vals = metrics_df["roc_auc"].fillna(0).values
        labels   = metrics_df["label"].values if "label" in metrics_df.columns else PATHOLOGY_LABELS

        colors = ["red" if v < 0.75 else "orange" if v < 0.80 else "steelblue"
                  for v in auc_vals]
        ax.bar(range(len(labels)), auc_vals, color=colors, alpha=0.85)
        ax.axhline(0.80, color="green", linestyle="--", linewidth=1.5, label="AUC=0.80 target")
        ax.axhline(0.75, color="orange", linestyle="--", linewidth=1, label="AUC=0.75 alert")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("ROC-AUC")
        ax.set_ylim([0, 1.05])
        ax.legend()
        ax.set_title("Per-Class ROC-AUC")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── F1 vs Radiologist comparison ──────────────────────
        if "radiologist_f1" in metrics_df.columns:
            st.markdown("**Model F1 vs Radiologist Benchmark:**")
            fig2, ax2 = plt.subplots(figsize=(14, 5))
            x = np.arange(len(labels))
            ax2.bar(x - 0.2, metrics_df["f1"].fillna(0), 0.4, label="Model F1", color="steelblue")
            ax2.bar(x + 0.2, metrics_df["radiologist_f1"].fillna(0), 0.4,
                    label="Radiologist F1", color="salmon", alpha=0.7)
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels, rotation=45, ha="right")
            ax2.set_ylabel("F1 Score")
            ax2.set_ylim([0, 1.05])
            ax2.legend()
            ax2.set_title("Model vs Radiologist — Per-Class F1")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

    except Exception as e:
        st.warning(f"Could not load metrics: {e}")
else:
    st.info("No metrics found. Run train_model.py first.")


# ============================================================
# SECTION 4 — PSI DRIFT MONITORING
# ============================================================

st.markdown("---")
st.subheader("📈 PSI Drift Monitoring")

if os.path.exists(PSI_PATH):
    try:
        df_psi = pd.read_csv(PSI_PATH)

        if "feature" in df_psi.columns and "drift_score" in df_psi.columns:
            # ── PSI summary table ─────────────────────────────
            df_psi["status"] = df_psi["drift_score"].apply(
                lambda x: "🔴 CRITICAL" if x >= PSI_HIGH
                else ("🟡 MODERATE" if x >= PSI_MODERATE else "✅ STABLE")
            )
            st.dataframe(
                df_psi[["feature", "drift_score", "status"]].sort_values("drift_score", ascending=False),
                use_container_width=True
            )

            # ── PSI bar chart ─────────────────────────────────
            fig, ax = plt.subplots(figsize=(14, 5))
            colors = [
                "red" if v >= PSI_HIGH else
                ("orange" if v >= PSI_MODERATE else "steelblue")
                for v in df_psi["drift_score"]
            ]
            ax.bar(range(len(df_psi)), df_psi["drift_score"], color=colors)
            ax.axhline(PSI_HIGH,     color="red",    linestyle="--", label=f"Critical PSI={PSI_HIGH}")
            ax.axhline(PSI_MODERATE, color="orange", linestyle="--", label=f"Moderate PSI={PSI_MODERATE}")
            ax.set_xticks(range(len(df_psi)))
            ax.set_xticklabels(df_psi["feature"], rotation=45, ha="right")
            ax.set_ylabel("PSI Score")
            ax.set_title("Feature/Score Drift — Population Stability Index")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    except Exception as e:
        st.warning(f"Could not load PSI report: {e}")
else:
    st.info("No PSI drift report found. Monitor scores will populate after inference.")


# ============================================================
# SECTION 5 — RECENT PREDICTIONS
# ============================================================

st.markdown("---")
st.subheader("📋 Recent Predictions Log")

if os.path.exists(LOG_PATH):
    try:
        df_log = pd.read_csv(LOG_PATH)

        # ── Summary stats ─────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Predictions", len(df_log))
        with col2:
            if "critical_finding" in df_log.columns:
                st.metric("Critical Findings", int(df_log["critical_finding"].sum()))
        with col3:
            if "uncertainty_flag" in df_log.columns:
                st.metric("Uncertain Predictions", int(df_log["uncertainty_flag"].sum()))

        # ── Recent predictions table ──────────────────────────
        cols_to_show = [c for c in [
            "timestamp", "identifier", "n_findings", "mean_probability",
            "critical_finding", "uncertainty_flag", "findings_str"
        ] if c in df_log.columns]

        st.dataframe(
            df_log.tail(50)[cols_to_show].iloc[::-1],
            use_container_width=True
        )

        # ── Findings distribution ─────────────────────────────
        if "findings_str" in df_log.columns:
            all_findings = []
            for s in df_log["findings_str"].dropna():
                all_findings += [x.strip() for x in s.split("|") if x.strip()]

            if all_findings:
                from collections import Counter
                counts = Counter(all_findings)
                top_n  = dict(counts.most_common(14))

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(list(top_n.keys()), list(top_n.values()), color="steelblue", alpha=0.8)
                ax.set_xticks(range(len(top_n)))
                ax.set_xticklabels(list(top_n.keys()), rotation=45, ha="right")
                ax.set_ylabel("Count")
                ax.set_title("Findings Distribution (All Predictions)")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    except Exception as e:
        st.warning(f"Could not load prediction log: {e}")
else:
    st.info("No prediction log found. Logs appear after first API predictions.")
