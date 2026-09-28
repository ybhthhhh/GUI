"""Export only trained stage-one compressor tensors, omitting optimizer state.

The original checkpoints remain on the shared volume.  The exported files are
plain safetensors, suitable for versioned baseline artifacts in GitHub LFS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--backend", choices=("qwen25vl", "llava_next", "internvl2"), required=True)
    args = parser.parse_args()

    import torch
    from safetensors.torch import save_file

    # weights_only refuses arbitrary pickled objects. Never load model code from
    # the checkpoint and never publish the optimizer or the training examples.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("compressor"), dict):
        raise ValueError("expected a stage-one checkpoint with compressor tensors")
    tensors = checkpoint["compressor"]
    if not tensors or not all(isinstance(value, torch.Tensor) for value in tensors.values()):
        raise ValueError("compressor contains no tensors or an unexpected value")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({key: value.detach().contiguous() for key, value in tensors.items()}, str(args.output))
    metadata = {
        "kind": "memw_stage1_compressor_baseline",
        "backend": args.backend,
        "checkpoint_step": int(checkpoint["step"]),
        "parameter_count": sum(value.numel() for value in tensors.values()),
        "tensor_count": len(tensors),
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "export_sha256": sha256(args.output),
        "export_bytes": args.output.stat().st_size,
        "format": "safetensors; compressor tensors only; no optimizer, backbone, dataset or secrets",
        "training_config": checkpoint.get("config", {}),
        "interpretation": "Model-specific Mem-W-style baseline; not a cross-transfer-compatible storyline encoder.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(json.dumps({key: metadata[key] for key in ("backend", "checkpoint_step", "parameter_count", "export_bytes", "export_sha256")}))


if __name__ == "__main__":
    main()
