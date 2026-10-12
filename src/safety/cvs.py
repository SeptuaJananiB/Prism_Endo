"""Critical View of Safety (CVS) assessment from a trained frame classifier (Endoscapes-CVS201 surgeon labels).

Model: ResNet-50 (dropout head) trained on a Colab T4 with outputs/cloud/cvs_fast2.py (320x576 input); test mAP 0.539
on 40 unseen Endoscapes videos (outputs/models/cvs_resnet50/cvs_report.json). It estimates whether each CVS criterion
(C1 hepatocystic triangle cleared, C2 lower gallbladder off the cystic plate, C3 two structures to the gallbladder) is
achieved - a "safe to clip?" indicator. It is NOT a tissue-injury risk and NOT clinically validated.
Scored every `every_s` seconds of video (CPU) and held in between.
"""

from __future__ import annotations

import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class CVSAssessor:
    def __init__(self, checkpoint, every_s: float = 1.0, height: int = 320, width: int = 576,
                 threshold: float = 0.5, threads: int | None = None):
        import torch
        import torchvision

        self.torch = torch
        if threads:
            torch.set_num_threads(int(threads))
        m = torchvision.models.resnet50(weights=None)
        m.fc = torch.nn.Sequential(torch.nn.Dropout(0.0), torch.nn.Linear(m.fc.in_features, 3))
        ck = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        m.load_state_dict(ck["model_state"])
        self.model = m.eval()
        self.every_s, self.h, self.w, self.threshold = float(every_s), int(height), int(width), float(threshold)
        self.last_t, self.last = None, None

    def predict(self, bgr: np.ndarray) -> np.ndarray:
        import cv2

        img = cv2.resize(bgr, (self.w, self.h), interpolation=cv2.INTER_AREA)
        x = ((img[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None].copy()
        with self.torch.no_grad():
            return self.torch.sigmoid(self.model(self.torch.from_numpy(x)))[0].numpy()

    def update(self, bgr: np.ndarray, t: float) -> dict:
        if self.last is None or self.last_t is None or t - self.last_t >= self.every_s:
            self.last, self.last_t = self.predict(bgr), t
        p = self.last
        return {"cvs_c1": round(float(p[0]), 3), "cvs_c2": round(float(p[1]), 3), "cvs_c3": round(float(p[2]), 3),
                "cvs_achieved": bool((p >= self.threshold).all())}
