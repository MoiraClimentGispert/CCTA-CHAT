"""Explicit, fingerprinted integration with a separate pinned CT-CHAT checkout."""
import importlib
from importlib.resources import files
import json
import os
from pathlib import Path
import subprocess
import sys
from .data import fingerprint

REVISION = "33c9b973bf1babb3b79b5ffd8ae45948e758a3f2"
# The integration uses PyTorch exclusively, including when TensorFlow is installed.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")


def git(checkout, *args):
    return subprocess.check_output(["git", "-C", str(checkout), *args], stderr=subprocess.STDOUT)


def integrate(checkout):
    checkout = Path(checkout).resolve()
    if git(checkout, "rev-parse", "HEAD").decode().strip() != REVISION:
        raise ValueError(f"CT-CHAT must be checked out at {REVISION}.")
    if git(checkout, "status", "--porcelain").strip():
        raise ValueError("Use a clean, separate upstream checkout; existing work will not be overwritten.")
    patch = files("ccta_chat").joinpath("integration/ctchat.patch").read_bytes()
    for args in (("apply", "--check", "--whitespace=nowarn", "-"), ("apply", "--whitespace=nowarn", "-")):
        subprocess.run(["git", "-C", str(checkout), *args], input=patch, check=True, capture_output=True)
    verify_backend(checkout)
    return {"upstream_revision": REVISION, "patch_applied": True}


def verify_backend(checkout):
    checkout = Path(checkout).resolve()
    if git(checkout, "rev-parse", "HEAD").decode().strip() != REVISION:
        raise ValueError("Wrong CT-CHAT revision; use the pinned checkout in docs/methods.md.")
    manifest = json.loads(files("ccta_chat").joinpath("integration/backend.json").read_text())
    for name, expected in manifest["files"].items():
        path = checkout / name
        if not path.is_file() or fingerprint(path) != expected:
            raise ValueError(f"Backend fingerprint mismatch: {name}. Apply the packaged integration patch to a clean checkout.")
    return checkout


def load_backend(checkout):
    checkout = verify_backend(checkout)
    for name, module in list(sys.modules.items()):
        if name == "llava" or name.startswith("llava."):
            origin = getattr(module, "__file__", None)
            if origin and not Path(origin).resolve().is_relative_to(checkout):
                raise ValueError("An unrelated llava package is already imported. Start a fresh Python process.")
    sys.path.insert(0, str(checkout))
    module = importlib.import_module("llava.model.language_model.llava_llama")
    if not Path(module.__file__).resolve().is_relative_to(checkout):
        raise ValueError("Backend import was shadowed by a different llava installation.")
    return module
