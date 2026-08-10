#!/usr/bin/env python
# ============================================================
# RUN API — Chest X-Ray Diagnosis ML System
# ============================================================
# Usage:
#   python scripts/run_api.py
#   python scripts/run_api.py --port 8080 --reload
#
# Endpoints:
#   GET  /          → home
#   GET  /health    → health check
#   GET  /model_info→ champion model metadata
#   GET  /labels    → 14 pathology labels + thresholds
#   POST /predict   → upload X-ray PNG → diagnosis
#   POST /predict_base64 → base64 image → diagnosis
# ============================================================

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import uvicorn


def parse_args():
    parser = argparse.ArgumentParser(description="Start Chest X-Ray Diagnosis FastAPI server")
    parser.add_argument("--host",   type=str,  default="0.0.0.0",   help="Host (default: 0.0.0.0)")
    parser.add_argument("--port",   type=int,  default=8000,         help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true",             help="Enable auto-reload (dev mode)")
    parser.add_argument("--workers",type=int,  default=1,            help="Number of workers (default: 1)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("CHEST X-RAY DIAGNOSIS API")
    print(f"  URL  : http://{args.host}:{args.port}")
    print(f"  Docs : http://{args.host}:{args.port}/docs")
    print("=" * 60)

    uvicorn.run(
        "serving.xray_api:app",
        host    = args.host,
        port    = args.port,
        reload  = args.reload,
        workers = args.workers if not args.reload else 1,
        log_level = "info",
    )


if __name__ == "__main__":
    main()
