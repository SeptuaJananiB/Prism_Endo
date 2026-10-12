"""Make DLC snapshots trained on DirectML loadable by DeepLabCut's safe (weights_only) loader.

DirectML tensors are pickled via torch._utils._rebuild_device_tensor_from_numpy, which torch.load(weights_only=True)
rejects. This loads OUR OWN snapshots (produced by scripts/dlc_train_gpu.py on this machine - trusted) with
weights_only=False, moves every tensor to CPU and re-saves. Each original is first copied to <name>.dml.bak.
Run with .venv-dml (the environment that wrote the files).

  .venv-dml\\Scripts\\python scripts\\dlc_snapshots_to_cpu.py <project>/dlc-models-pytorch
"""

import shutil
import sys
from pathlib import Path

import torch
import torch_directml  # noqa: F401  (needed to unpickle DirectML tensors)


def to_cpu(o):
    if isinstance(o, torch.Tensor):
        return o.detach().cpu()
    if isinstance(o, dict):
        return {k: to_cpu(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(to_cpu(v) for v in o)
    return o


def main(argv=None):
    root = Path((argv if argv is not None else sys.argv[1:])[0])
    for snap in sorted(root.rglob("snapshot-*.pt")):
        bak = snap.with_suffix(".pt.dml.bak")
        if not bak.exists():
            shutil.copy2(snap, bak)
        # Trusted, locally produced file (see docstring); full unpickling is required for DirectML tensors.
        data = torch.load(str(bak), map_location="cpu", weights_only=False)
        torch.save(to_cpu(data), str(snap))
        torch.load(str(snap), map_location="cpu", weights_only=True)  # verify the safe loader now accepts it
        print(f"converted {snap.name} (backup {bak.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
