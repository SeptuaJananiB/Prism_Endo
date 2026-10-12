"""Train a DeepLabCut (PyTorch engine) model on an AMD/Intel GPU via DirectML.

DeepLabCut does not know DirectML; importing torch_directml registers the 'privateuseone' device,
which is then passed to deeplabcut.train_network. Requires deeplabcut in .venv-dml.
Training is refused unless the project has enough human labels (dlc_workflow.require_labels).

  .venv-dml\\Scripts\\python scripts\\dlc_train_gpu.py --config <project>/config.yaml --epochs 50
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch_directml  # noqa: E402,F401  (registers the DirectML device)

from src.tracking import dlc_workflow as wf  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--save-epochs", type=int, default=5)
    ap.add_argument("--device", default=None, help="default: DirectML device")
    ap.add_argument("--snapshot", default=None, help="continue from these weights (e.g. snapshot-019.pt); "
                    "DLC restarts epoch numbering, so back up the train folder first")
    a = ap.parse_args(argv)
    wf.require_labels(a.config)
    device = a.device or str(torch_directml.device())
    if a.snapshot and device.startswith("privateuseone"):
        # torch_directml cannot be used as torch.load(map_location=...) (TypeError inside torch_directml.device);
        # load snapshot weights on CPU instead - DLC moves the model to the DirectML device afterwards.
        import torch

        _load = torch.load

        def _cpu_load(*args, **kwargs):
            if "privateuseone" in str(kwargs.get("map_location", "")):
                kwargs["map_location"] = "cpu"
            return _load(*args, **kwargs)

        torch.load = _cpu_load
    t0 = time.perf_counter()
    import deeplabcut

    deeplabcut.train_network(a.config, shuffle=1, epochs=a.epochs, save_epochs=a.save_epochs,
                             batch_size=a.batch_size, device=device, snapshot_path=a.snapshot)
    print(f"TRAINING_DONE device={device} epochs={a.epochs} seconds={time.perf_counter() - t0:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
