#!/usr/bin/env python
# ============================================================
# RUN SIMULATION — Chest X-Ray Diagnosis ML System
# ============================================================
# Usage:
#   python scripts/run_simulation.py
#   python scripts/run_simulation.py --scenario critical --n 50
#   python scripts/run_simulation.py --scenario all --n 30
#
# Scenarios:
#   normal   → healthy patients, ~90% no-finding
#   critical → emergency intake, Pneumonia/Pneumothorax heavy
#   mixed    → realistic hospital outpatient mix
#   all      → runs all 3 scenarios sequentially
# ============================================================

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate X-Ray API traffic")
    parser.add_argument(
        "--scenario", type=str, default="mixed",
        choices=["normal", "critical", "mixed", "all"],
        help="Simulation scenario (default: mixed)"
    )
    parser.add_argument(
        "--n", type=int, default=20,
        help="Number of X-rays to simulate per scenario (default: 20)"
    )
    parser.add_argument(
        "--api-url", type=str, default="http://localhost:8000",
        help="FastAPI server URL (default: http://localhost:8000)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["XRAY_API_URL"] = args.api_url

    print("=" * 60)
    print("CHEST X-RAY SIMULATOR")
    print(f"  Scenario  : {args.scenario}")
    print(f"  N per run : {args.n}")
    print(f"  API URL   : {args.api_url}")
    print("=" * 60)

    from simulation.xray_simulator import simulate_xrays, simulate_all_scenarios

    if args.scenario == "all":
        simulate_all_scenarios(n_each=args.n)
    else:
        simulate_xrays(n=args.n, scenario=args.scenario)


if __name__ == "__main__":
    main()
