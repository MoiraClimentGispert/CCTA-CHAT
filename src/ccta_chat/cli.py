"""Public entry points. Heavy dependencies are imported only by optional commands."""
import argparse
import json
import subprocess
import sys
from .data import read_json, write_json, prepare


def parser():
    root = argparse.ArgumentParser(prog="ccta-chat", description="CCTA-CHAT preparation, model integration and evaluation.")
    root.add_argument("--version", action="version", version="CCTA-CHAT 0.1.0")
    commands = root.add_subparsers(dest="command", required=True)
    def evaluation(name, description):
        p = commands.add_parser(name, help=description)
        p.add_argument("--predictions", required=True)
        p.add_argument("--references", required=True)
        return p
    p = evaluation("evaluate", "Score ID-aligned answers with token-set P/R/F1 (CPU, no extras).")
    p.add_argument("--json", action="store_true", help="Print machine-readable metrics.")
    p = evaluation("coco", "Score BLEU-1, ROUGE-L and CIDEr (coco extra).")
    p = evaluation("semantic", "Compute nine ROUGE/token/BERT columns (semantic extra).")
    p.add_argument("--bert-model", required=True, help="Existing local RoBERTa-large model directory.")
    p.add_argument("--legacy-record", action="store_true", help="Explicit positional whole-record historical schema.")
    p = commands.add_parser("prepare", help="Create marked CT-CHAT conversations from curated QA.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("generate-qa", help="Generate five QA pairs per article with a local model (model extra).")
    p.add_argument("--input", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("retrieve-articles", help="Retrieve PMC article text and IDs (requires network).")
    p.add_argument("--query", required=True)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--split", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("integrate", help="Apply the packaged patch to a clean pinned CT-CHAT checkout.")
    p.add_argument("--checkout", required=True)
    p = commands.add_parser("merge-base", help="Merge pretrained CT-CHAT LoRA into a full base for FT1 (model extra).")
    for arg in ("backend", "base", "tokenizer", "projector", "adapter", "output"):
        p.add_argument("--"+arg, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    p = commands.add_parser("encode", help="Encode one NIfTI using explicit CT-ViT source/weights (imaging extra).")
    for name in ("input", "output", "encoder-source", "encoder-weights"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--spacing", type=float, nargs=3, metavar=("X", "Y", "Z"), help="Override header spacing in mm.")
    p.add_argument("--reuse", action="store_true")
    p.add_argument("--device", default="cuda")
    def model_command(name, description):
        p = commands.add_parser(name, help=description)
        for arg in ("backend", "base", "tokenizer", "projector", "features", "output"):
            p.add_argument("--"+arg, required=True)
        p.add_argument("--adapter")
        p.add_argument("--device", default="cuda")
        p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
        return p
    p = model_command("train", "Train LoRA + projector using cached features (model extra).")
    for name in ("config", "data", "base-provenance"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--resume", help="Explicit Trainer checkpoint directory; independent of adapter initialization.")
    p = model_command("infer", "Generate marked image or text-only answers (model extra).")
    p.add_argument("--input", required=True)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p = evaluation("judge", "Judge aligned answers with a local LLM (model extra).")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    return root


def main(argv=None):
    root = parser()
    args = root.parse_args(argv)
    try:
        if args.command == "evaluate":
            from .evaluation import evaluate
            predictions, references = read_json(args.predictions), read_json(args.references)
            result = evaluate(predictions, references)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print("CCTA-CHAT | answer-only evaluation")
                print(f"Records: {result['records']} | Answers: {result['answers']} | Excluded: {result['excluded']}")
                print(f"Token precision: {result['token_precision']:.6f}")
                print(f"Token recall:    {result['token_recall']:.6f}")
                print(f"Token F1:        {result['token_f1']:.6f}")
                print("Aggregation: macro-average over turns (lowercase whitespace-token sets)")
                if all(r.get("synthetic") is True for r in references + predictions):
                    print("Synthetic fixture: illustrative; not thesis results.")
            return 0
        if args.command == "prepare":
            result = prepare(read_json(args.input))
            write_json(args.output, result)
            result = {"records": len(result), "image_records": sum(bool(r["image"]) for r in result)}
        elif args.command == "integrate":
            from .backend import integrate
            result = integrate(args.checkout)
        elif args.command == "encode":
            from .encoding import encode
            result = encode(args.input, args.output, args.encoder_source, args.encoder_weights,
                            spacing=args.spacing, reuse=args.reuse, device=args.device)
        elif args.command == "train":
            from .training import train
            result = train(args)
        elif args.command == "infer":
            from .model import infer
            result = infer(args)
        elif args.command == "merge-base":
            from .model import merge_base
            result = merge_base(args)
        elif args.command == "generate-qa":
            from .generation import generate_qa
            result = generate_qa(args)
        elif args.command == "retrieve-articles":
            from .data import retrieve_articles
            rows = retrieve_articles(args.query, args.limit, args.split)
            write_json(args.output, rows)
            result = {"articles": len(rows)}
        elif args.command == "judge":
            from .generation import judge
            result = judge(args)
        elif args.command == "coco":
            from .evaluation import coco_scores
            result = coco_scores(read_json(args.predictions), read_json(args.references))
        elif args.command == "semantic":
            from pathlib import Path
            if not Path(args.bert_model).is_dir():
                raise ValueError("--bert-model must be an existing local model directory.")
            from .evaluation import semantic_scores
            result = semantic_scores(read_json(args.predictions), read_json(args.references),
                                     legacy=args.legacy_record, bert_model=args.bert_model)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, KeyError, OSError, ImportError, subprocess.CalledProcessError) as exc:
        if isinstance(exc, ImportError):
            message = f"Optional dependency unavailable: {exc}. Install the extra documented in docs/methods.md."
        else:
            message = str(exc)
        print(f"ccta-chat: {message}", file=sys.stderr)
        return 2
