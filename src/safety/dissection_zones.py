"""Suggested dissection zones for laparoscopic cholecystectomy (Go / No-Go), rule-based from tissue segmentation.

Basis in the literature (see docs/DISSECTION_ZONES.md):
  * GoNoGoNet (Madani et al., Ann Surg 2022): green "Go" overlay = safe zone of dissection inside the hepatocystic
    triangle, red "No-Go" overlay = deeper area where the common bile duct / hepatic structures run.
  * SAGES Safe Cholecystectomy / Critical View of Safety (Strasberg): clear the fibro-fatty tissue of the hepatocystic
    triangle, staying on the gallbladder; stay above Rouviere's sulcus / the R4U line (away from the bile duct).

GoNoGoNet learned the zones from expert annotations. We have no such labels, so this module APPROXIMATES them from the
tissue classes our model segments reliably:
  Go     = fat / connective tissue within `go_band_frac` of the gallbladder edge (the tissue to clear, on the
           gallbladder), at least `danger_margin_frac` away from any danger structure.
  No-Go  = danger structures (cystic duct, hepatic vein, blood, bowel) plus a margin, and fibro-fatty tissue further
           than `deep_frac` from the gallbladder (the deep area where the bile duct runs).
It is a visual aid for reviewing video, not surgical guidance: Rouviere's sulcus and the bile duct are not
segmented, and the zones have not been compared with expert annotations.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DEFAULTS = {
    "enabled": True,
    "go_band_frac": 0.06,          # Go band width from the gallbladder edge (fraction of image diagonal)
    "deep_frac": 0.16,             # fibro-fatty tissue beyond this from the gallbladder is No-Go (deep zone)
    "danger_margin_frac": 0.03,    # margin around danger structures
    "min_gallbladder_frac": 0.01,  # zones only when the gallbladder covers at least this fraction of the image
    "dissect_classes": ["fat", "connective_tissue"],
    "danger_classes": ["cystic_duct", "hepatic_vein", "blood", "gastrointestinal_tract"],
}

GO_BGR = (80, 200, 60)
NOGO_BGR = (40, 40, 230)


@dataclass
class Zones:
    go: np.ndarray | None = None       # bool mask
    nogo: np.ndarray | None = None     # bool mask
    available: bool = False
    reason: str = ""

    def zone_at(self, x: float, y: float) -> str:
        """'go', 'no_go' or '' for a point (e.g. an instrument tip)."""
        if not self.available:
            return ""
        h, w = self.go.shape
        xi, yi = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
        return "no_go" if self.nogo[yi, xi] else "go" if self.go[yi, xi] else ""


def compute(tissue_labels: np.ndarray | None, class_names: list[str], cfg: dict | None = None) -> Zones:
    c = {**DEFAULTS, **(cfg or {})}
    if not c.get("enabled", True) or tissue_labels is None:
        return Zones(reason="disabled" if tissue_labels is not None else "no tissue segmentation")
    if "gallbladder" not in class_names:
        return Zones(reason="tissue model has no gallbladder class")
    h, w = tissue_labels.shape[:2]
    diag = float(np.hypot(h, w))
    gb = tissue_labels == class_names.index("gallbladder")
    if gb.mean() < c["min_gallbladder_frac"]:
        return Zones(reason="gallbladder not visible")

    def mask(names):
        m = np.zeros((h, w), bool)
        for n in names:
            if n in class_names:
                m |= tissue_labels == class_names.index(n)
        return m

    d_gb = cv2.distanceTransform((~gb).astype(np.uint8), cv2.DIST_L2, 3)       # px to the gallbladder
    danger = mask(c["danger_classes"])
    d_danger = cv2.distanceTransform((~danger).astype(np.uint8), cv2.DIST_L2, 3) if danger.any() \
        else np.full((h, w), np.inf, np.float32)
    dissect = mask(c["dissect_classes"])

    near_danger = d_danger <= c["danger_margin_frac"] * diag
    go = dissect & (d_gb <= c["go_band_frac"] * diag) & ~near_danger
    nogo = near_danger | (dissect & (d_gb > c["deep_frac"] * diag))
    go &= ~nogo
    return Zones(go=go, nogo=nogo, available=True)


def draw(img: np.ndarray, z: Zones, alpha: float = 0.28) -> np.ndarray:
    """Green Go / red No-Go overlay with outlines and a small legend (GoNoGoNet-style colours)."""
    if not z.available or (not z.go.any() and not z.nogo.any()):
        return img
    out = img.copy()
    for m, col in ((z.go, GO_BGR), (z.nogo, NOGO_BGR)):
        if m.any():
            layer = out.copy()
            layer[m] = col
            out = cv2.addWeighted(layer, alpha, out, 1 - alpha, 0)
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, col, 1, cv2.LINE_AA)
    h = out.shape[0]
    y = h - 30
    for label, col in (("Go zone (estimate)", GO_BGR), ("No-Go zone (estimate)", NOGO_BGR)):
        cv2.rectangle(out, (8, y - 9), (18, y + 1), col, -1)
        cv2.putText(out, label, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        y += 15
    return out
