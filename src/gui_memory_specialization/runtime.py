"""Node-local runtime compatibility helpers.

The 31571 Torch build lives in a vendor path that also exposes an incompatible
``flash_attn`` package. During the single Transformers import, this module masks
only that package's discoverability so attention falls back to the standard
PyTorch implementation. No shared package is altered.
"""

from __future__ import annotations

import importlib.util


def import_torch_then_transformers():
    import torch
    original_find_spec = importlib.util.find_spec

    def without_incompatible_vendor_packages(name: str, *args, **kwargs):
        # DeepSpeed is not used for a frozen single-GPU forward pass and its
        # vendor build is incompatible with the installed Pydantic version.
        if name in {"flash_attn", "deepspeed"}:
            return None
        return original_find_spec(name, *args, **kwargs)

    # Model classes are lazily imported after ``transformers`` itself, so this
    # process-local override must remain installed for the scorer's lifetime.
    # The scorer runs in a short-lived subprocess; no global environment changes.
    importlib.util.find_spec = without_incompatible_vendor_packages
    import transformers

    return torch, transformers
