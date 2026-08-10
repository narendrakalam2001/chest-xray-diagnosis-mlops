# ============================================================
# X-RAY SIMULATOR — Chest X-Ray Diagnosis ML System
# ============================================================
# Simulates real-world X-ray API traffic for 3 scenarios:
#   1. normal    — healthy patients, no pathology expected
#   2. critical  — high-risk patients (Pneumonia, Pneumothorax)
#   3. mixed     — realistic hospital intake mix
# ============================================================

import os
import io
import time
import random
import logging
import base64
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── API URL ───────────────────────────────────────────────────
API_URL = os.getenv("XRAY_API_URL", "http://localhost:8000")

PATHOLOGY_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

# Scenario configs: label prevalence weights per scenario
SCENARIO_CONFIG = {
    "normal": {
        "description":     "Healthy patient — no significant pathology",
        "no_finding_prob": 0.90,
        "pathology_probs": {lbl: 0.01 for lbl in PATHOLOGY_LABELS},
    },
    "critical": {
        "description":     "High-risk patients — emergency intake",
        "no_finding_prob": 0.05,
        "pathology_probs": {
            "Pneumonia":       0.40,
            "Pneumothorax":    0.30,
            "Effusion":        0.35,
            "Atelectasis":     0.25,
            "Cardiomegaly":    0.20,
            "Mass":            0.15,
            "Nodule":          0.15,
            "Consolidation":   0.20,
            "Edema":           0.20,
            "Emphysema":       0.10,
            "Infiltration":    0.30,
            "Fibrosis":        0.08,
            "Pleural_Thickening": 0.10,
            "Hernia":          0.03,
        },
    },
    "mixed": {
        "description":     "Realistic hospital outpatient mix",
        "no_finding_prob": 0.55,
        "pathology_probs": {
            "Atelectasis":     0.10,
            "Cardiomegaly":    0.08,
            "Effusion":        0.12,
            "Infiltration":    0.18,
            "Mass":            0.05,
            "Nodule":          0.05,
            "Pneumonia":       0.07,
            "Pneumothorax":    0.04,
            "Consolidation":   0.06,
            "Edema":           0.06,
            "Emphysema":       0.04,
            "Fibrosis":        0.04,
            "Pleural_Thickening": 0.04,
            "Hernia":          0.02,
        },
    },
}


# ============================================================
# SYNTHETIC X-RAY IMAGE GENERATOR
# ============================================================

def generate_synthetic_xray(
    scenario:    str  = "mixed",
    image_size:  tuple= (224, 224),
    seed:        int  = None,
) -> bytes:
    """
    Generate a synthetic grayscale chest X-ray-like image.
    Used for simulation when real images are not available.

    Produces realistic-looking gray gradient image with:
        - Rib-like horizontal lines (simulated anatomy)
        - Noise texture (simulates X-ray grain)
        - Optional bright spots (simulated pathologies)

    Returns image as PNG bytes.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    W, H = image_size

    # ── Base gradient (lung field appearance) ────────────────
    base = np.zeros((H, W), dtype=np.float32)

    # Oval lung fields (left + right)
    for cx, cy, rx, ry in [(W * 0.33, H * 0.5, W * 0.22, H * 0.38),
                            (W * 0.67, H * 0.5, W * 0.22, H * 0.38)]:
        yy, xx = np.ogrid[:H, :W]
        mask = ((xx - cx) ** 2 / rx ** 2 + (yy - cy) ** 2 / ry ** 2) <= 1
        base[mask] = np.random.uniform(0.6, 0.85)

    # Mediastinum
    base[:, int(W * 0.4):int(W * 0.6)] = np.clip(
        base[:, int(W * 0.4):int(W * 0.6)] * 0.5 + 0.1, 0, 1
    )

    # ── Noise texture ────────────────────────────────────────
    noise = np.random.normal(0, 0.04, (H, W)).astype(np.float32)
    base  = np.clip(base + noise, 0, 1)

    # ── Rib lines ────────────────────────────────────────────
    for rib_y in range(int(H * 0.25), int(H * 0.75), int(H * 0.08)):
        base[rib_y:rib_y+2, :] = np.clip(base[rib_y:rib_y+2, :] * 0.70, 0, 1)

    # ── Pathology simulation (scenario-specific) ─────────────
    cfg = SCENARIO_CONFIG.get(scenario, SCENARIO_CONFIG["mixed"])
    if random.random() > cfg["no_finding_prob"]:
        for lbl, prob in cfg["pathology_probs"].items():
            if random.random() < prob:
                _add_pathology_marker(base, lbl, H, W)

    # ── Convert to PIL RGB (3 channel for model) ─────────────
    img_uint8 = (base * 255).astype(np.uint8)
    img_rgb   = np.stack([img_uint8] * 3, axis=-1)
    pil_img   = Image.fromarray(img_rgb)
    pil_img   = pil_img.filter(ImageFilter.GaussianBlur(radius=0.5))

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def _add_pathology_marker(
    base:     np.ndarray,
    label:    str,
    H:        int,
    W:        int,
) -> None:
    """Add subtle visual marker for a specific pathology."""

    if label in ("Pneumonia", "Consolidation", "Infiltration"):
        # Diffuse opacity in one lung field
        cx = random.choice([int(W * 0.28), int(W * 0.72)])
        cy = int(H * random.uniform(0.35, 0.65))
        r  = int(H * random.uniform(0.08, 0.15))
        yy, xx = np.ogrid[:H, :W]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        base[mask] = np.clip(base[mask] * random.uniform(0.5, 0.75), 0, 1)

    elif label in ("Nodule", "Mass"):
        # Small dense spot
        cx = random.choice([int(W * 0.28), int(W * 0.72)])
        cy = int(H * random.uniform(0.3, 0.7))
        r  = int(H * (0.03 if label == "Nodule" else 0.06))
        yy, xx = np.ogrid[:H, :W]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        base[mask] = np.clip(base[mask] * 0.4, 0, 1)

    elif label == "Pneumothorax":
        # Bright rim at lung edge
        side = random.choice(["left", "right"])
        if side == "left":
            base[int(H*0.2):int(H*0.8), int(W*0.05):int(W*0.15)] = 0.9
        else:
            base[int(H*0.2):int(H*0.8), int(W*0.85):int(W*0.95)] = 0.9

    elif label == "Cardiomegaly":
        # Wider mediastinum
        base[:, int(W*0.35):int(W*0.65)] = np.clip(
            base[:, int(W*0.35):int(W*0.65)] * 0.45, 0, 1
        )

    elif label == "Effusion":
        # Blunting of costophrenic angle (bottom of lung)
        base[int(H*0.75):, int(W*0.1):int(W*0.45)] = np.clip(
            base[int(H*0.75):, int(W*0.1):int(W*0.45)] * 0.5, 0, 1
        )


# ============================================================
# SEND TO API
# ============================================================

def send_xray(image_bytes: bytes, idx: int, scenario: str) -> dict:
    """Send synthetic X-ray to API and print result."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            files={"file": (f"sim_{idx:04d}.png", image_bytes, "image/png")},
            timeout=30,
        )
        if response.status_code == 200:
            result   = response.json()
            findings = [f["label"] for f in result.get("findings", [])]
            critical = "🔴 CRITICAL" if result.get("critical_finding") else "✅"
            uncertain = "⚠️ UNCERTAIN" if result.get("uncertainty_flag") else ""
            logger.info(
                "[%3d] scenario=%-8s  findings=%s  %s %s",
                idx + 1, scenario,
                findings if findings else ["None"],
                critical, uncertain
            )
            return result
        else:
            logger.warning("[%3d] API error: HTTP %d", idx + 1, response.status_code)
            return {}
    except requests.exceptions.ConnectionError:
        logger.warning("[%3d] API not reachable at %s — is the server running?", idx + 1, API_URL)
        return {}
    except Exception as e:
        logger.warning("[%3d] Request failed: %s", idx + 1, e)
        return {}


# ============================================================
# RUN SIMULATION
# ============================================================

def simulate_xrays(n: int = 20, scenario: str = "mixed") -> None:
    """
    Simulate n X-ray predictions for a given scenario.

    Args:
        n:        number of X-rays to simulate
        scenario: one of 'normal', 'critical', 'mixed'
    """
    cfg = SCENARIO_CONFIG.get(scenario, SCENARIO_CONFIG["mixed"])
    print(f"\nSimulating {n} X-rays  |  scenario={scenario}")
    print(f"  Description: {cfg['description']}")
    print("-" * 65)

    results = []
    for i in range(n):
        img_bytes = generate_synthetic_xray(scenario=scenario, seed=i)
        result    = send_xray(img_bytes, i, scenario)
        results.append(result)
        time.sleep(0.3)

    # ── Summary ───────────────────────────────────────────────
    n_findings  = sum(1 for r in results if r.get("n_findings", 0) > 0)
    n_critical  = sum(1 for r in results if r.get("critical_finding", False))
    n_uncertain = sum(1 for r in results if r.get("uncertainty_flag",  False))

    print("-" * 65)
    print(f"Simulation complete  |  n={n}")
    print(f"  With findings  : {n_findings} / {n}  ({100*n_findings/max(n,1):.0f}%)")
    print(f"  Critical       : {n_critical} / {n}  ({100*n_critical/max(n,1):.0f}%)")
    print(f"  Uncertain      : {n_uncertain} / {n}  ({100*n_uncertain/max(n,1):.0f}%)")


def simulate_all_scenarios(n_each: int = 10) -> None:
    """Run simulation for all 3 scenarios."""
    for scenario in ["normal", "critical", "mixed"]:
        simulate_xrays(n=n_each, scenario=scenario)
        time.sleep(1.0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="X-Ray Simulator")
    parser.add_argument("--scenario", default="mixed",
                        choices=["normal", "critical", "mixed", "all"])
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    if args.scenario == "all":
        simulate_all_scenarios(n_each=args.n)
    else:
        simulate_xrays(n=args.n, scenario=args.scenario)
