"""Compact LoRA/projector trainer derived from the selected outer training path.

One record is one training example. All assistant spans are supervised; prompts
are masked. Frozen CT-ViT runs offline. Resume is always an explicit option.
"""
from pathlib import Path
from .data import read_json, write_json, fingerprint
from .backend import REVISION
from .model import load_model, initialize_adapter, trainable_components, load_feature, artifact_fingerprints, load_projector


def encode_conversation(record, tokenizer, maximum):
    import torch
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
    conv = conv_templates["llama3"].copy()
    spans = []
    conversation = record.get("conversations", [])
    if not conversation or len(conversation) % 2:
        raise ValueError("Training requires alternating prepared human/gpt messages.")
    image = record.get("image")
    markers = sum(m["value"].count("<image>") for m in conversation if m["from"] == "human")
    if markers != int(bool(image)):
        raise ValueError("Training records need exactly one image marker when image is set, and zero otherwise.")
    def tokens():
        return tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX)
    for i in range(0, len(conversation), 2):
        human, answer = conversation[i:i+2]
        if human["from"] != "human" or answer["from"] != "gpt":
            raise ValueError("Training messages must alternate human and gpt.")
        conv.append_message(conv.roles[0], human["value"])
        conv.append_message(conv.roles[1], None)
        prefix = tokens()
        conv.messages[-1][-1] = answer["value"]
        complete = tokens()
        if complete[:len(prefix)] != prefix:
            raise ValueError("Tokenizer merges across the answer boundary; use the matching CT-CHAT tokenizer.")
        spans.append((len(prefix), len(complete)))
    cap = maximum - (255 if image else 0)  # image index becomes 256 projected queries
    if cap <= 0:
        raise ValueError("model_max_length must leave room for 256 visual tokens and the answer.")
    ids = torch.tensor(complete[:cap], dtype=torch.long)
    labels = torch.full_like(ids, IGNORE_INDEX)
    for start, end in spans:
        labels[start:min(end, len(ids))] = ids[start:min(end, len(ids))]
    if not labels.ne(IGNORE_INDEX).any():
        raise ValueError(f"No answer survives truncation for {record.get('record_id')}; increase model_max_length.")
    return {"input_ids": ids, "labels": labels}


def collate(instances, pad_token_id):
    import torch
    from torch.nn.utils.rnn import pad_sequence
    ids = pad_sequence([r["input_ids"] for r in instances], batch_first=True, padding_value=pad_token_id)
    labels = pad_sequence([r["labels"] for r in instances], batch_first=True, padding_value=-100)
    # Length masks preserve genuine EOS tokens when EOS also serves as pad.
    mask = torch.arange(ids.shape[1])[None] < torch.tensor([len(r["input_ids"]) for r in instances])[:, None]
    result = {"input_ids": ids, "labels": labels, "attention_mask": mask}
    actual = [r["image"] for r in instances if "image" in r]
    if actual:
        if any(r.shape != actual[0].shape for r in actual):
            raise ValueError("Cached feature shapes in a batch differ; use a uniform encoder profile.")
        # One slot per record, including a zero placeholder for text-only records.
        result["images"] = torch.stack([r.get("image", torch.zeros_like(actual[0])) for r in instances])
    return result


def train(args):
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments
    settings = read_json(args.config)
    mode = settings["mode"]
    if mode not in {"ft1", "ft2"}:
        raise ValueError("Training mode must be ft1 or ft2.")
    if mode == "ft2" and not args.adapter:
        raise ValueError("FT2 requires --adapter; the adapter is loaded trainably, independently of warmup/resume.")
    provenance = read_json(args.base_provenance)
    if mode == "ft1" and provenance.get("kind") != "ctchat-merged-base":
        raise ValueError("FT1 requires a base provenance JSON with kind='ctchat-merged-base'.")
    if provenance.get("config_sha256") != fingerprint(Path(args.base)/"config.json"):
        raise ValueError("Base provenance config_sha256 does not match --base/config.json.")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Training output is nonempty; choose a new directory or explicit --resume checkpoint.")
    model, tokenizer = load_model(args.backend, args.base, args.tokenizer, args.projector,
                                  device=args.device, dtype=args.dtype)
    model = initialize_adapter(model, mode=mode, settings=settings, adapter=args.adapter)
    components = trainable_components(model)
    if args.resume:
        checkpoint_projector = Path(args.resume)/"non_lora_trainables.bin"
        if not checkpoint_projector.is_file():
            raise ValueError("Resume checkpoint needs non_lora_trainables.bin for the trained projector.")
        load_projector(model, checkpoint_projector)
    model.config.use_cache = False
    model.config.tokenizer_model_max_length = settings["model_max_length"]
    model.config.tokenizer_padding_side = "right"
    records = read_json(args.data)
    if not records:
        raise ValueError("Training data is empty.")
    class Dataset(torch.utils.data.Dataset):
        def __len__(self): return len(records)
        def __getitem__(self, i):
            item = encode_conversation(records[i], tokenizer, settings["model_max_length"])
            if records[i].get("image"):
                item["image"] = load_feature(Path(args.features)/records[i]["image"])
            return item
    training_args = TrainingArguments(
        output_dir=str(output), num_train_epochs=settings["epochs"], learning_rate=settings["learning_rate"],
        warmup_ratio=settings["warmup_ratio"], lr_scheduler_type=settings["scheduler"],
        per_device_train_batch_size=settings["batch_size"], gradient_accumulation_steps=settings["gradient_accumulation"],
        bf16=args.dtype == "bfloat16", fp16=args.dtype == "float16", remove_unused_columns=False,
        save_strategy="epoch", report_to=[], use_cpu=args.device == "cpu", seed=42)
    manifest = {"initialization": mode, "base_provenance": provenance, "settings": settings,
                "upstream_revision": REVISION, "trainable_parameters": components,
                "data_sha256": fingerprint(args.data), "projector_sha256": fingerprint(args.projector),
                "base_artifacts": artifact_fingerprints(args.base), "tokenizer_artifacts": artifact_fingerprints(args.tokenizer),
                "adapter_artifacts": artifact_fingerprints(args.adapter) if args.adapter else None,
                "trainer_resume": args.resume,
                "resume_artifacts": artifact_fingerprints(args.resume) if args.resume else None,
                "effective_training_arguments": training_args.to_dict()}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"initialization.json", manifest)
    def save_projector(directory):
        torch.save({name: value.detach().cpu() for name, value in model.named_parameters() if "mm_projector" in name},
                   Path(directory)/"non_lora_trainables.bin")
    class ProjectorCheckpoint(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if args.should_save:
                save_projector(Path(args.output_dir)/f"checkpoint-{state.global_step}")
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, train_dataset=Dataset(),
                      callbacks=[ProjectorCheckpoint()],
                      data_collator=lambda items: collate(items, tokenizer.pad_token_id))
    # Transformers 4.44 saves NumPy RNG arrays. Recent PyTorch releases require
    # their concrete reconstruction types in a scoped weights-only allowlist.
    from contextlib import nullcontext
    context = nullcontext()
    if args.resume and hasattr(torch.serialization, "safe_globals"):
        import numpy as np
        from numpy.core.multiarray import _reconstruct
        context = torch.serialization.safe_globals([_reconstruct, np.ndarray, np.dtype, type(np.dtype("uint32"))])
    with context:
        trainer.train(resume_from_checkpoint=args.resume or None)
    trainer.save_state()
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    save_projector(output)
    return {"mode": mode, "trainable_parameters": components, "epochs": trainer.state.epoch}
