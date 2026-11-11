"""Explicit CT-CHAT adapter/projector initialization and ID-preserving inference.

Reconstructs the selected outer loader/trainer and nested validation writer.
Shared LLaMA/projector machinery lives in the fingerprinted upstream checkout.
"""
from pathlib import Path
from .backend import load_backend, REVISION
from .data import fingerprint, read_json, write_json, validate_records


def local_directory(path, name):
    path = Path(path).resolve()
    if not path.is_dir():
        raise ValueError(f"{name} must be an existing local artifact directory.")
    return path


def artifact_fingerprints(directory):
    """Hash local model/tokenizer artifacts without deserializing them."""
    return {p.name: fingerprint(p) for p in sorted(Path(directory).iterdir())
            if p.is_file() and p.suffix in {".json", ".bin", ".safetensors", ".model", ".txt"}}


def adapter_settings(path, expected_rank=None):
    path = local_directory(path, "Adapter")
    config = read_json(path / "adapter_config.json")
    if config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM":
        raise ValueError("Expected a CAUSAL_LM LoRA adapter_config.json.")
    if expected_rank is not None and config.get("r") != expected_rank:
        raise ValueError(f"Adapter rank {config.get('r')} differs from requested {expected_rank}; "
                         "use its saved rank or supply an explicitly converted adapter.")
    if not any((path / n).is_file() for n in ("adapter_model.safetensors", "adapter_model.bin")):
        raise ValueError("Adapter weights are missing (adapter_model.safetensors or adapter_model.bin).")
    return config


def load_projector(model, path):
    import torch
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(str(path))
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    # Normalize the prefix exactly at mm_projector, regardless of PEFT nesting.
    state = {k.split("mm_projector.", 1)[1]: value for k, value in state.items() if "mm_projector." in k}
    expected = model.get_model().mm_projector.state_dict()
    if state.keys() != expected.keys():
        raise ValueError(f"Projector key coverage differs: {len(expected.keys()-state.keys())} missing, "
                         f"{len(state.keys()-expected.keys())} unexpected.")
    for key in state:
        if state[key].shape != expected[key].shape:
            raise ValueError(f"Projector tensor shape differs for {key}.")
    model.get_model().mm_projector.load_state_dict(state, strict=True)


def initialize_adapter(model, *, mode, settings, adapter=None):
    from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
    if mode == "ft1":
        if adapter:
            raise ValueError("FT1 takes an already merged CT-CHAT base and creates a fresh CAD adapter.")
        config = LoraConfig(r=settings["lora_rank"], lora_alpha=settings["lora_alpha"],
                            lora_dropout=settings["lora_dropout"], bias="none", task_type="CAUSAL_LM",
                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        return get_peft_model(model, config)
    if mode != "ft2" or not adapter:
        raise ValueError("FT2 requires --adapter pointing to an existing trained CT-CHAT adapter.")
    adapter_settings(adapter, settings.get("lora_rank"))
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True, local_files_only=True)
    from peft.utils.save_and_load import load_peft_weights
    supplied = load_peft_weights(str(adapter), device="cpu", local_files_only=True)
    restored = get_peft_model_state_dict(model)
    if supplied.keys() != restored.keys():
        raise ValueError("Adapter tensor keys differ from the instantiated model; loading was not complete.")
    for key in supplied:
        loaded = restored[key].detach().cpu()
        if supplied[key].shape != loaded.shape or not loaded.equal(supplied[key].to(loaded.dtype)):
            raise ValueError(f"Adapter tensor did not restore exactly: {key}.")
    return model


def trainable_components(model):
    model.requires_grad_(False)
    counts = {"lora": 0, "projector": 0}
    for name, param in model.named_parameters():
        kind = "lora" if "lora_" in name else "projector" if "mm_projector" in name else None
        if kind:
            param.requires_grad_(True)
            counts[kind] += param.numel()
    if not all(counts.values()):
        raise ValueError("Both LoRA and projector must have trainable parameters.")
    return counts


def load_model(backend, base, tokenizer_path, projector, *, device="cpu", dtype="float32"):
    import torch
    from transformers import AutoTokenizer
    module = load_backend(backend)
    base = local_directory(base, "Base")
    tokenizer_path = local_directory(tokenizer_path, "Tokenizer")
    raw = read_json(base / "config.json")
    if (base / "adapter_config.json").exists():
        raise ValueError("--base must contain full base weights, not a PEFT adapter directory.")
    raw.pop("model_type", None)
    config = module.LlavaConfig(**raw)
    config.mm_vision_tower = "cached-ctvit"
    config.mm_hidden_size = 512
    config.mm_context_size = 512
    config.mm_projector_type = "attn_pool+mlp2x_gelu"
    config.mm_patch_merge_type = "flat"
    config.image_aspect_ratio = "square"
    model = module.LlavaLlamaForCausalLM.from_pretrained(
        str(base), config=config, torch_dtype=getattr(torch, dtype), local_files_only=True)
    load_projector(model, projector)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=False, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    for token in ("<short_answer>", "<long_answer>", "<multiple_choice>", "<report_generation>"):
        if token not in tokenizer.get_vocab():
            raise ValueError(f"Tokenizer lacks {token}; supply the matching CT-CHAT tokenizer/config.")
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        raise ValueError("Tokenizer length differs from base embedding rows; provide matching artifacts.")
    return model.to(device), tokenizer


def load_feature(path):
    import numpy as np
    import torch
    with np.load(path, allow_pickle=False) as data:
        arr = data["arr"]
    if arr.ndim != 5 or arr.shape[0] != 1 or arr.shape[-1] != 512 or not np.isfinite(arr).all():
        raise ValueError("Cached arr must be finite with shape [1,Z,X,Y,512].")
    return torch.from_numpy(arr[0].copy())


def generate_turns(model, tokenizer, records, *, features, max_new_tokens=256):
    """The CT-CHAT LLaMA generate override returns new tokens from inputs_embeds."""
    import torch
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX
    validate_records(records, answers=False)
    output = []
    for record in records:
        image = record.get("image")
        tensor = load_feature(Path(features) / image).unsqueeze(0).to(model.device, dtype=model.dtype) if image else None
        conv = conv_templates["llama3"].copy()
        turns = []
        for i, turn in enumerate(record["turns"]):
            task = record.get("task", "short_answer")
            if task not in {"short_answer", "long_answer", "multiple_choice", "report_generation"}:
                raise ValueError(f"Unsupported task {task!r}.")
            if "<image>" in turn["question"] or "<provided>" in turn["question"]:
                raise ValueError("Inference accepts unmarked questions and inserts image markers itself.")
            prefix = "<image>\n<provided>\n" if image and i == 0 else ""
            conv.append_message(conv.roles[0], f"{prefix}<{task}> {turn['question']}")
            conv.append_message(conv.roles[1], None)
            ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")[None].to(model.device)
            with torch.inference_mode():
                generated = model.generate(ids, attention_mask=torch.ones_like(ids), images=tensor,
                                           do_sample=False, max_new_tokens=max_new_tokens,
                                           pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            answer = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
            if not answer:
                raise ValueError(f"Empty generation for {record['record_id']}/{turn['turn_id']}.")
            conv.messages[-1][-1] = answer
            turns.append({"turn_id": turn["turn_id"], "question": turn["question"], "answer": answer})
        output.append({"record_id": record["record_id"], "scan_id": record["scan_id"], "turns": turns})
    return output


def infer(args):
    model, tokenizer = load_model(args.backend, args.base, args.tokenizer, args.projector,
                                  device=args.device, dtype=args.dtype)
    if args.adapter:
        from peft import PeftModel
        adapter_settings(args.adapter)
        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True).merge_and_unload()
    model.eval()
    results = generate_turns(model, tokenizer, read_json(args.input), features=args.features,
                             max_new_tokens=args.max_new_tokens)
    write_json(args.output, results)
    write_json(str(args.output) + ".run.json", {"upstream": REVISION, "input_sha256": fingerprint(args.input),
               "base_artifacts": artifact_fingerprints(args.base),
               "tokenizer_artifacts": artifact_fingerprints(args.tokenizer),
               "projector_sha256": fingerprint(args.projector),
               "adapter_artifacts": artifact_fingerprints(args.adapter) if args.adapter else None,
               "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens}})
    return {"records": len(results), "answers": sum(len(r["turns"]) for r in results)}


def merge_base(args):
    """Export CT-CHAT's adapter into a full base for a fresh FT1 CAD adapter."""
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Merge output is nonempty; choose a new directory.")
    model, tokenizer = load_model(args.backend, args.base, args.tokenizer, args.projector,
                                  device=args.device, dtype=args.dtype)
    model = initialize_adapter(model, mode="ft2", settings={"lora_rank": None}, adapter=args.adapter)
    model = model.merge_and_unload()
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    write_json(output/"base-provenance.json", {"kind": "ctchat-merged-base", "upstream_revision": REVISION,
               "config_sha256": fingerprint(output/"config.json"),
               "base_artifacts": artifact_fingerprints(args.base),
               "adapter_artifacts": artifact_fingerprints(args.adapter),
               "projector_sha256": fingerprint(args.projector),
               "tokenizer_artifacts": artifact_fingerprints(args.tokenizer)})
    return {"kind": "ctchat-merged-base", "config_sha256": fingerprint(output/"config.json")}
