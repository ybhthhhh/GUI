"""Run one supervised visual-action LoRA update on Qwen2.5-VL.

The purpose is to prove that the selected node can execute a screenshot-aware
forward pass, backward pass, and optimizer step.  It is deliberately a single
synthetic observation and must not be reported as a training or benchmark run.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def patch_vendor_flash_attention() -> None:
    """Bridge the node vendor's legacy rotary symbol before Transformers loads."""
    try:
        import flash_attn.layers.rotary as rotary
    except ImportError:
        return
    if not hasattr(rotary, "apply_rotary_emb") and hasattr(rotary, "apply_rotary_emb_func"):
        rotary.apply_rotary_emb = rotary.apply_rotary_emb_func


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--action", default="click(x=640, y=360)")
    args = parser.parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)

    patch_vendor_flash_attention()
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Qwen2.5-VL training smoke")
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(args.device)
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.train()
    messages = [{"role": "user", "content": [
        {"type": "image", "path": str(args.image)},
        {"type": "text", "text": "Inspect the GUI screenshot and output the next action."},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    inputs = {name: value.to(args.device) for name, value in inputs.items()}
    action_ids = processor.tokenizer(
        " " + args.action, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(args.device)
    inputs["input_ids"] = torch.cat((inputs["input_ids"], action_ids), dim=1)
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    labels = torch.full_like(inputs["input_ids"], -100)
    labels[:, -action_ids.shape[1] :] = action_ids
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    loss = model(**inputs, labels=labels).loss
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite loss: {loss.item()}")
    loss.backward()
    optimizer.step()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    peak_mib = torch.cuda.max_memory_allocated(args.device) // (1024 * 1024)
    print("QWEN25VL_LORA_TRAIN_SMOKE_OK", f"loss={loss.item():.6f}", f"trainable={trainable}", f"peak_mib={peak_mib}")


if __name__ == "__main__":
    main()
