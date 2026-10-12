import numpy as np
import pytest

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")

from src.safety.cvs import CVSAssessor  # noqa: E402


@pytest.fixture()
def ckpt(tmp_path):
    m = torchvision.models.resnet50(weights=None)
    m.fc = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(m.fc.in_features, 3))
    p = tmp_path / "best.pt"
    torch.save({"model_state": m.state_dict(), "epoch": 1, "val": {"mAP": 0.5}}, p)
    return p


def test_outputs_three_criteria_and_flag(ckpt):
    a = CVSAssessor(ckpt, height=64, width=96)
    out = a.update(np.full((120, 200, 3), 100, np.uint8), 0.0)
    assert set(out) == {"cvs_c1", "cvs_c2", "cvs_c3", "cvs_achieved"}
    assert all(0.0 <= out[k] <= 1.0 for k in ("cvs_c1", "cvs_c2", "cvs_c3"))
    assert out["cvs_achieved"] == all(out[k] >= 0.5 for k in ("cvs_c1", "cvs_c2", "cvs_c3"))


def test_scores_only_every_interval(ckpt):
    a = CVSAssessor(ckpt, every_s=1.0, height=64, width=96)
    calls = []
    real = a.predict
    a.predict = lambda img: calls.append(1) or real(img)
    frame = np.full((120, 200, 3), 50, np.uint8)
    for t in (0.0, 0.2, 0.5, 0.99, 1.0, 1.5, 2.1):
        a.update(frame, t)
    assert len(calls) == 3  # t = 0.0, 1.0, 2.1
