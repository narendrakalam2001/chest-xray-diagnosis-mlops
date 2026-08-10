#!/usr/bin/env python
# ============================================================
# RUN DASHBOARD — Chest X-Ray Diagnosis ML System
# ============================================================
# Usage:
#   python scripts/run_dashboard.py
#   python scripts/run_dashboard.py --port 8502
#   python scripts/run_dashboard.py --api-url http://localhost:8000
#
# Dashboard sections:
#   1. Real-Time Alerts
#   2. Champion vs Challenger History
#   3. KPIs + Charts (per-class AUC, F1 vs radiologist)
#   4. PSI Drift Monitoring
#   5. Recent Predictions Log
# ============================================================

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description="Launch Chest X-Ray Monitoring Dashboard")
    parser.add_argument("--port",    type=int, default=8501,
                        help="Streamlit port (default: 8501)")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000",
                        help="FastAPI server URL (default: http://localhost:8000)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Inject API URL for dashboard to pick up
    os.environ["XRAY_API_URL"] = args.api_url

    dashboard_path = os.path.join(
        os.path.dirname(__file__), "..", "monitoring", "monitoring_dashboard.py"
    )
    dashboard_path = os.path.abspath(dashboard_path)

    print("=" * 60)
    print("CHEST X-RAY MONITORING DASHBOARD")
    print(f"  Dashboard : http://localhost:{args.port}")
    print(f"  API URL   : {args.api_url}")
    print("=" * 60)

    cmd = [
        "streamlit", "run", dashboard_path,
        "--server.port",       str(args.port),
        "--server.address",    "0.0.0.0",
        "--theme.base",        "light",
        "--browser.gatherUsageStats", "false",
    ]

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
