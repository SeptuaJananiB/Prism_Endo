"""Compute-device selection and logging helpers."""

from __future__ import annotations

import logging
import os


def get_logger(name: str = "prism") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers and not logger.handlers:
        logging.basicConfig(
            level=os.environ.get("PRISM_LOGLEVEL", "INFO"),
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    return logger


def directml_device() -> str | None:
    """DirectML device string (AMD/Intel/NVIDIA GPUs on Windows via the
    `torch-directml` package), or None if it is not installed/usable."""
    try:
        import torch_directml

        if torch_directml.device_count() > 0:
            return str(torch_directml.device())  # e.g. "privateuseone:0"
    except Exception:
        pass
    return None


def select_device(requested: str = "auto") -> str:
    """Pick a device: 'auto' prefers CUDA, then DirectML, then CPU.
    Explicit values: 'cpu', 'cuda', 'dml'."""
    import torch

    requested = (requested or "auto").lower()
    if requested == "cpu":
        return "cpu"
    if requested in ("auto", "cuda") or requested.startswith("cuda"):
        if torch.cuda.is_available():
            return "cuda"
    if requested in ("auto", "dml", "directml"):
        dml = directml_device()
        if dml:
            return dml
    if requested != "auto":
        get_logger().warning("Device '%s' requested but not available; falling back to CPU.", requested)
    return "cpu"


def is_directml(device: str) -> bool:
    return str(device).startswith("privateuseone")


def describe_hardware() -> dict:
    import platform

    info = {"platform": platform.platform(), "python": platform.python_version(),
            "cpu_count": os.cpu_count()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
        try:
            import torch_directml

            if torch_directml.device_count():
                info["directml_gpu"] = torch_directml.device_name(0)
        except Exception:
            pass
        info["torch_threads"] = torch.get_num_threads()
    except ImportError:
        info["torch"] = None
    return info
