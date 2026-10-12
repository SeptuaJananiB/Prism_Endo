"""Draw masks, keypoints, trajectories, proximity and risk onto a frame."""

from __future__ import annotations

import cv2
import numpy as np

from src.segmentation.predictor import PALETTE_BGR

RISK_COLORS = {"Minimal": (80, 170, 60), "Low": (0, 200, 220), "Medium": (0, 140, 255),
               "High": (0, 0, 230), "Unavailable": (128, 128, 128)}
KP_COLORS = [(255, 0, 255), (255, 255, 0), (0, 255, 0), (255, 128, 0)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, org, scale=0.5, color=(255, 255, 255), thick=1):
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def _target(img, p, color):
    cv2.circle(img, p, 9, color, 1, cv2.LINE_AA)
    cv2.circle(img, p, 6, color, 1, cv2.LINE_AA)
    cv2.circle(img, p, 3, color, -1, cv2.LINE_AA)


def _legend(img, items):
    """Semi-transparent legend box, top-right: colour swatch + name per drawn class."""
    if not items:
        return
    h, w = img.shape[:2]
    bw, row = 130, 16
    x0, y0 = w - bw - 6, 32
    box = img[y0:y0 + row * len(items) + 8, x0:x0 + bw]
    box[:] = (box * 0.45).astype(np.uint8)
    for i, (name, color) in enumerate(items):
        y = y0 + 6 + i * row
        if name == "tip":
            _target(img, (x0 + 13, y + 6), color)
        else:
            cv2.rectangle(img, (x0 + 6, y + 1), (x0 + 20, y + 11), color, -1)
        _text(img, name, (x0 + 28, y + 11), 0.4)


def blend_mask(img, mask, color, alpha):
    if mask is None or not mask.any():
        return
    img[mask] = (img[mask] * (1 - alpha) + np.asarray(color) * alpha).astype(np.uint8)


def draw_frame(frame, *, inst_labels=None, inst_names=None, tissue_labels=None, tissue_names=None,
               tissue_draw=None, keypoints=(), motions=None, trails=None, proximities=None,
               risks=None, frame_risk=None, status_lines=(), alpha=0.4, kp_class=None,
               target_marks=None) -> np.ndarray:
    """kp_class maps keypoint name -> instrument class name, so a tip is drawn in the
    same colour as its instrument's mask."""
    img = frame.copy()
    h, w = img.shape[:2]
    legend = []
    # Tissue regions: light fill + boundary contour, labelled.
    if tissue_labels is not None and tissue_names:
        for name in tissue_draw or []:
            cid = tissue_names.index(name)
            m = tissue_labels == cid
            if not m.any():
                continue
            color = PALETTE_BGR[cid % len(PALETTE_BGR)]
            blend_mask(img, m, color, alpha * 0.5)
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, color, 1, cv2.LINE_AA)
            big = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(big) > 0.02 * h * w:  # leader-line label for each sizeable region
                mm = cv2.moments(big)
                c = (int(mm["m10"] / max(mm["m00"], 1)), int(mm["m01"] / max(mm["m00"], 1)))
                if cv2.pointPolygonTest(big, c, False) < 0:  # centroid outside a curved region: use a point inside
                    c = tuple(big[len(big) // 2][0])
                legend.append((name.replace("_", " "), color))
                cv2.circle(img, c, 3, color, -1, cv2.LINE_AA)
                side = -1 if c[0] > w * 0.6 else 1
                elbow = (c[0] + side * 18, c[1] - 18)
                end = (elbow[0] + side * 30, elbow[1])
                cv2.line(img, c, elbow, color, 1, cv2.LINE_AA)
                cv2.line(img, elbow, end, color, 1, cv2.LINE_AA)
                lab = name.replace("_", " ")
                tw = cv2.getTextSize(lab, FONT, 0.4, 1)[0][0]
                _text(img, lab, (end[0] + 3 if side > 0 else end[0] - tw - 3, end[1] + 4), 0.4, color)
    # Instrument masks.
    if inst_labels is not None and inst_names:
        for cid, name in enumerate(inst_names):
            if cid == 0:
                continue
            m = inst_labels == cid
            blend_mask(img, m, KP_COLORS[(cid - 1) % len(KP_COLORS)], alpha)
            if m.any():
                legend.insert(0, (name.replace("_", "-"), KP_COLORS[(cid - 1) % len(KP_COLORS)]))
    # Keypoints, trajectories, velocity, proximity lines.
    for i, kp in enumerate(keypoints):
        cls = (kp_class or {}).get(kp.name)
        if inst_names and cls in inst_names and inst_names.index(cls) > 0:
            color = KP_COLORS[(inst_names.index(cls) - 1) % len(KP_COLORS)]
        else:
            color = KP_COLORS[i % len(KP_COLORS)]
        if trails and trails.get(kp.name) and len(trails[kp.name]) > 1:
            pts = np.array(trails[kp.name][-12:], np.int32)  # short, thin trail; skip tracking jumps
            n = len(pts)
            for j in range(1, n):
                if np.hypot(*(pts[j] - pts[j - 1])) > 40:
                    continue
                c = tuple(int(v * (0.3 + 0.7 * j / n)) for v in color)
                cv2.line(img, tuple(pts[j - 1]), tuple(pts[j]), c, 1, cv2.LINE_AA)
        if kp.x is None or kp.y is None:
            continue
        p = (int(round(kp.x)), int(round(kp.y)))
        short = {"lhook_tip": "L-hook", "grasper_tip": "Grasper", "grasper_tip_2": "Grasper 2"}.get(kp.name, kp.name)
        if kp.valid:
            _target(img, p, (0, 0, 230))  # small red target: dot + two thin rings
        else:  # low confidence: thin grey ring only, not used downstream
            cv2.circle(img, p, 5, (170, 170, 170), 1, cv2.LINE_AA)
            continue
        # leader line: diagonal then horizontal, label at the end (keeps text off the tip)
        side = -1 if p[0] > w * 0.6 else 1
        elbow = (p[0] + side * 22, p[1] - 22)
        end = (elbow[0] + side * 40, elbow[1])
        cv2.line(img, p, elbow, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.line(img, elbow, end, (230, 230, 230), 1, cv2.LINE_AA)
        tw = cv2.getTextSize(short, FONT, 0.45, 1)[0][0]
        _text(img, short, (end[0] + 4 if side > 0 else end[0] - tw - 4, end[1] + 4), 0.45)
        pr = (proximities or {}).get(kp.name)
        if pr is not None and pr.status == "ok" and pr.closest_x is not None and not pr.inside:
            q = (int(pr.closest_x), int(pr.closest_y))
            cv2.line(img, p, q, (0, 220, 255), 1, cv2.LINE_AA)
            _text(img, f"{pr.distance_px:.0f}px", ((p[0] + q[0]) // 2 + 4, (p[1] + q[1]) // 2), 0.38, (0, 220, 255))
    # Risk banner.
    cat = frame_risk.category if frame_risk is not None else "Unavailable"
    color = RISK_COLORS.get(cat, (128, 128, 128))
    cv2.rectangle(img, (0, 0), (w, 26), color, -1)
    if frame_risk is not None:
        _text(img, f"Risk: {cat}", (6, 18), 0.55)
    else:
        _text(img, "Risk: n/a (no tip)", (6, 18), 0.55)
    _legend(img, legend + [("tip", (0, 0, 230))])
    return img


def _wrap(s: str, n: int) -> list[str]:
    words, lines, cur = s.split(), [], ""
    for wd in words:
        if len(cur) + len(wd) + 1 > n:
            lines.append(cur)
            cur = wd
        else:
            cur = f"{cur} {wd}".strip()
    if cur:
        lines.append(cur)
    return lines[:3]
