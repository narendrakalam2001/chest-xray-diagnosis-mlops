#!/usr/bin/env python
# ============================================================
# TRAIN MODEL — Chest X-Ray Diagnosis ML System
# ============================================================
# Usage:
#   python scripts/train_model.py
#   python scripts/train_model.py --backbone resnet50
#   python scripts/train_model.py --backbone efficientnet_b0 --epochs 20
# ============================================================

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/training.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Chest X-Ray Diagnosis model (NIH ChestX-ray14)"
    )
    parser.add_argument(
        "--backbone", type=str, default="densenet121",
        choices=["densenet121", "resnet50", "efficientnet_b0"],
        help="CNN backbone to train (default: densenet121)"
    )
    parser.add_argument(
        "--images-dir", type=str, default=None,
        help="Path to directory containing X-ray PNG images"
    )
    parser.add_argument(
        "--labels-csv", type=str, default=None,
        help="Path to Data_Entry_2017.csv"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Optional epoch override
    if args.epochs:
        import src.config as cfg
        cfg.EPOCHS = args.epochs
        logger.info("Epochs overridden to %d", args.epochs)

    logger.info("=" * 60)
    logger.info("CHEST X-RAY DIAGNOSIS — TRAINING STARTED")
    logger.info("  Backbone : %s", args.backbone)
    logger.info("=" * 60)

    from src.training_pipeline import run_training

    model, summary, thresholds = run_training(
        backbone    = args.backbone,
        images_dir  = args.images_dir,
        labels_csv  = args.labels_csv,
    )

    logger.info("Training complete.")
    logger.info("  Mean AUC : %.4f", summary.get("mean_auc", 0))
    logger.info("  Mean F1  : %.4f", summary.get("mean_f1",  0))


if __name__ == "__main__":
    main()
