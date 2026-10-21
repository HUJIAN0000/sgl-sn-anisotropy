"""Small, deterministic I/O helpers."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.generic):
        return jsonable(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, Path):
        return str(x)
    return x


def write_json(path, obj):
    """Atomic replacement prevents partially written checkpoints/reports."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     delete=False, suffix=".tmp") as f:
        json.dump(jsonable(obj), f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
        temporary = f.name
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(jsonable(obj), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def code_hash():
    files = sorted((ROOT / "aa_revision").glob("*.py"))
    return fingerprint({p.name: sha256(p) for p in files})


def environment():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ["numpy", "scipy", "pandas", "matplotlib", "emcee", "h5py", "Jinja2"]:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not installed"
    return result
