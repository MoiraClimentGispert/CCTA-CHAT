"""Offline CCTA preprocessing reconstructed from the nested encoder.

Uses NIfTI-scaled HU once, header spacing by default, no orientation canonicalizing.
The external CT-ViT source must be supplied explicitly, and is fingerprinted.
"""
import importlib
import math
from pathlib import Path
import sys
from .data import fingerprint, read_json, write_json


def preprocess(volume, spacing, *, target_spacing=(0.35, 0.35, 0.30), shape=(480, 480, 240)):
    import numpy as np
    import torch
    from torch.nn import functional as F
    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError("Expected one finite 3D volume, in scaled Hounsfield units.")
    if any(not math.isfinite(s) or s <= 0 for s in (*spacing, *target_spacing)):
        raise ValueError("Voxel spacing must be finite and positive in x,y,z order.")
    # Original voxel axes are retained; the encoder receives [B,C,Z,X,Y].
    x = torch.as_tensor(np.clip(volume, -100, 800).astype(np.float32)).permute(2, 0, 1)[None, None]
    size = tuple(max(1, int(volume.shape[i] * spacing[i] / target_spacing[i])) for i in (2, 0, 1))
    x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)[0, 0]
    x = (x + 100) / 900
    target = tuple(shape[i] for i in (2, 0, 1))
    starts = [max(0, (n - t) // 2) for n, t in zip(x.shape, target)]
    x = x[tuple(slice(s, s + t) for s, t in zip(starts, target))]
    pads = []
    for n, t in reversed(list(zip(x.shape, target))):
        before = (t - n) // 2
        pads.extend((before, t - n - before))
    return F.pad(x, pads, value=-1)[None, None]


def encode(input_path, output, encoder_source, encoder_weights, *, spacing=None, reuse=False, device="cuda"):
    import numpy as np
    import nibabel as nib
    import torch
    source = Path(encoder_source).resolve()
    module_path = source / "transformer_maskgit" / "__init__.py"
    if not module_path.is_file():
        raise ValueError("--encoder-source must contain transformer_maskgit/__init__.py (CT-CLIP transformer_maskgit directory).")
    source_hashes = {str(p.relative_to(source)): fingerprint(p) for p in sorted(source.rglob("*.py"))}
    image = nib.load(str(input_path))
    voxel_spacing = tuple(float(value) for value in (spacing or image.header.get_zooms()[:3]))
    manifest = {"schema": 1, "input_sha256": fingerprint(input_path), "encoder_sha256": fingerprint(encoder_weights),
                "encoder_source": source_hashes, "implementation_sha256": fingerprint(__file__),
                "input_spacing_xyz_mm": list(voxel_spacing), "spacing_policy": "explicit" if spacing else "nifti-header",
                "target_spacing_xyz_mm": [0.35, 0.35, 0.30], "window_hu": [-100, 800],
                "normalization": "(HU+100)/900", "crop_xyz": [480, 480, 240], "padding": -1,
                "orientation": "native NIfTI voxel axes", "intensity_scaling": "nibabel get_fdata applies NIfTI slope/intercept once"}
    output = Path(output)
    sidecar = output.with_suffix(".provenance.json")
    if output.exists():
        if not reuse:
            raise ValueError("Output exists; choose a new path or --reuse with matching provenance.")
        saved = read_json(sidecar) if sidecar.exists() else {}
        if saved.get("preprocessing") != manifest or saved.get("output_sha256") != fingerprint(output):
            raise ValueError("Cached embedding provenance differs or is absent; choose a new output path.")
        return {"reused": True, "sha256": saved["output_sha256"]}
    loaded = sys.modules.get("transformer_maskgit")
    if loaded and not Path(loaded.__file__).resolve().is_relative_to(source):
        raise ValueError("A different transformer_maskgit is loaded; use a fresh process.")
    sys.path.insert(0, str(source))
    module = importlib.import_module("transformer_maskgit")
    if not Path(module.__file__).resolve().is_relative_to(source):
        raise ValueError("External CT-ViT import was shadowed.")
    encoder = module.CTViT(dim=512, codebook_size=8192, image_size=480, patch_size=20,
                         temporal_patch_size=10, spatial_depth=4, temporal_depth=4, dim_head=32, heads=8)
    encoder.load(str(encoder_weights))
    encoder = encoder.to(device).eval()
    tensor = preprocess(image.get_fdata(dtype=np.float32), voxel_spacing).to(device)
    with torch.inference_mode():
        arr = encoder(tensor, return_encoded_tokens=True).cpu().numpy()
    if arr.ndim != 5 or arr.shape[0] != 1 or arr.shape[-1] != 512:
        raise ValueError(f"Unexpected CT-ViT output {arr.shape}; expected [1,Z,X,Y,512].")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        np.savez_compressed(stream, arr=arr)
    write_json(sidecar, {"preprocessing": manifest, "shape": list(arr.shape), "output_sha256": fingerprint(output)})
    return {"reused": False, "shape": list(arr.shape), "sha256": fingerprint(output)}
