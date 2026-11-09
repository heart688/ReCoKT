import csv
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import torch


def _musa(required=False):
    """Register the Moore Threads backend before resolving a MUSA device."""
    try:
        import torch_musa
    except ImportError as error:
        if required:
            raise RuntimeError("MUSA requested but torch_musa could not be loaded") from error
        return None
    return torch_musa


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if _musa() is not None and torch.musa.is_available():
        torch.musa.manual_seed_all(seed)


def choose_device(name="auto"):
    name = str(name)
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda:0"
        elif _musa() is not None and torch.musa.is_available():
            name = "musa:0"
        else:
            name = "cpu"
    if name.startswith("musa"):
        _musa(required=True)
        if not torch.musa.is_available():
            raise RuntimeError("MUSA requested but unavailable; CPU fallback is disabled")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; CPU fallback is disabled")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    device = torch.device(name)
    if device.type in ("cuda", "musa"):
        backend = getattr(torch, device.type)
        index = device.index if device.index is not None else 0
        if index >= backend.device_count():
            raise RuntimeError(f"Requested {device.type}:{index} is not visible")
        backend.set_device(index)
        device = torch.device(device.type, index)
    return device


@contextmanager
def fork_device_rng(seed, device):
    """Isolate CPU and the selected accelerator RNG for resumable local fits."""
    device = choose_device(device)
    cpu_state = torch.get_rng_state()
    backend = getattr(torch, device.type) if device.type in ("cuda", "musa", "mps") else None
    accelerator_state = (backend.get_rng_state(device.index)
                         if device.type in ("cuda", "musa")
                         else backend.get_rng_state() if backend is not None else None)
    try:
        torch.random.default_generator.manual_seed(seed)
        if backend is not None:
            backend.manual_seed(seed)
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if device.type in ("cuda", "musa"):
            backend.set_rng_state(accelerator_state, device.index)
        elif backend is not None:
            backend.set_rng_state(accelerator_state)


def device_metadata(device):
    device = choose_device(device)
    result = {"resolved_device": str(device), "device_type": device.type,
              "accelerator": device.type in ("cuda", "musa", "mps"),
              "torch_version": str(torch.__version__),
              "default_dtype": str(torch.get_default_dtype()),
              "autocast_enabled": torch.is_autocast_enabled()}
    if device.type in ("cuda", "musa"):
        backend = getattr(torch, device.type)
        properties = backend.get_device_properties(device.index)
        result.update(device_name=backend.get_device_name(device.index),
                      visible_device_count=backend.device_count(),
                      total_memory_bytes=int(properties.total_memory),
                      visible_devices=os.environ.get("MUSA_VISIBLE_DEVICES" if device.type == "musa"
                                                     else "CUDA_VISIBLE_DEVICES"))
    if device.type == "musa":
        result["torch_musa_version"] = str(getattr(_musa(required=True), "__version__", "unknown"))
        mudnn = getattr(torch.backends, "mudnn", None)
        result["mudnn_allow_tf32"] = getattr(mudnn, "allow_tf32", None)
    return result


def reset_peak_memory(device):
    device = choose_device(device)
    if device.type in ("cuda", "musa"):
        synchronize(device)
        getattr(torch, device.type).reset_peak_memory_stats(device.index)


def memory_snapshot(device):
    device = choose_device(device)
    result = {"device": str(device), "scope": "selected device allocator in this process"}
    if device.type in ("cuda", "musa"):
        synchronize(device)
        backend = getattr(torch, device.type)
        result.update(allocated_bytes=int(backend.memory_allocated(device.index)),
                      reserved_bytes=int(backend.memory_reserved(device.index)),
                      peak_allocated_bytes=int(backend.max_memory_allocated(device.index)),
                      peak_reserved_bytes=int(backend.max_memory_reserved(device.index)))
    return result


def environment():
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except subprocess.CalledProcessError:
        revision = None
    source_root = Path(__file__).parent
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(source_root.glob("*.py"))}
    musa = _musa()
    musa_available = musa is not None and torch.musa.is_available()
    return {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "torch": str(torch.__version__),
        "numpy": np.__version__, "cuda_available": torch.cuda.is_available(),
        "cuda_build": torch.version.cuda,
        "cuda_devices": [torch.cuda.get_device_name(i)
                         for i in range(torch.cuda.device_count())],
        "musa_available": musa_available,
        "musa_devices": ([torch.musa.get_device_name(i)
                          for i in range(torch.musa.device_count())] if musa_available else []),
        "torch_musa_version": str(getattr(musa, "__version__", "unavailable")),
        "mps_available": torch.backends.mps.is_available(),
        "git_revision": revision, "source_sha256": source_hashes, "pid": os.getpid(),
    }


def json_dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid4().hex[:8])
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def csv_dump(path, rows):
    path = Path(path)
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def new_run(root, name):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(root) / (name + "_" + stamp + "_" + uuid4().hex[:6])
    path.mkdir(parents=True, exist_ok=False)
    return path


def configuration_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def move_batch(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def state_copy(model):
    return {name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()}


def parameter_bytes(model):
    return sum(t.numel() * t.element_size() for t in model.state_dict().values())


def save_checkpoint(path, model, metadata):
    torch.save({"state_dict": state_copy(model), "model": model.metadata(),
                "metadata": metadata}, path)


def synchronize(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "musa":
        _musa(required=True)
        torch.musa.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
