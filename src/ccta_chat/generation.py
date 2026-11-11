"""Local-model literature QA and a separate answer judge, with saved provenance."""
import json
from pathlib import Path
from .data import article_prompt, read_json, write_json, fingerprint
from .evaluation import align


def pipeline_for(path):
    from . import backend  # select the PyTorch-only integration before Transformers imports
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    path = Path(path)
    if not path.is_dir():
        raise ValueError("Supply an existing local model directory; model downloads are not implicit.")
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True, device_map="auto")
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def generate_qa(args):
    articles = read_json(args.input)
    generate = pipeline_for(args.model)
    result = []
    seen = set()
    for article in articles:
        identity = article.get("article_id")
        if not identity or identity in seen or not article.get("split"):
            raise ValueError("Articles need unique article_id and explicit split.")
        seen.add(identity)
        prompt = article_prompt(article["title"], article["body"])
        text = generate([{"role": "user", "content": prompt}], max_new_tokens=1024,
                        do_sample=False, return_full_text=False)[0]["generated_text"]
        pairs = json.loads(text)
        if not isinstance(pairs, list) or len(pairs) != 5:
            raise ValueError(f"Article {identity}: expected a JSON list of five QA pairs; inspect generation separately.")
        result.append({"record_id": f"article-{identity}", "scan_id": "text-only", "source": "literature",
                       "article_id": identity, "split": article["split"], "task": "long_answer", "image": None,
                       "turns": [{"turn_id": str(i), "question": pair["question"], "answer": pair["answer"]}
                                 for i, pair in enumerate(pairs)]})
    from .data import validate_records
    validate_records(result)
    write_json(args.output, result)
    write_json(str(args.output)+".run.json", {"input_sha256": fingerprint(args.input),
               "model_config_sha256": fingerprint(Path(args.model)/"config.json"),
               "body_limit_characters": 4000, "pairs_per_article": 5, "do_sample": False})
    return {"articles": len(result), "answers": 5*len(result)}


def judge(args):
    predictions, references = read_json(args.predictions), read_json(args.references)
    pairs = align(predictions, references)
    questions = {(r["record_id"], t["turn_id"]): t["question"] for r in references for t in r["turns"]}
    generate = pipeline_for(args.model)
    rows = []
    for key, answer, reference in pairs:
        prompt = ("Compare the candidate answer with the reference for the supplied question. "
                  "Score correctness and completeness from 0 to 10. Return only JSON with score and rationale.\n" +
                  json.dumps({"question": questions[key], "reference": reference, "candidate": answer}))
        raw = generate([{"role": "user", "content": prompt}], max_new_tokens=256,
                       do_sample=False, return_full_text=False)[0]["generated_text"]
        parsed = json.loads(raw)
        value = parsed.get("score")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 10:
            raise ValueError(f"Judge returned invalid score for {key!r}; no aggregate was produced.")
        rows.append({"record_id": key[0], "turn_id": key[1], "question": questions[key],
                     "reference": reference, "answer": answer, "judge": parsed})
    write_json(args.output, {"model_config_sha256": fingerprint(Path(args.model)/"config.json"),
                           "protocol": "release-judge", "triplets": rows,
                           "mean": sum(r["judge"]["score"] for r in rows)/len(rows)})
    return {"answers": len(rows), "protocol": "release-judge"}
