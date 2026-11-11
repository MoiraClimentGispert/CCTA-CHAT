"""Answer-only alignment and historical whitespace-token-set P/R/F1.

The legacy semantic extraction follows the nested P_R_F1_metrics.py. COCO metric
definitions follow new_llm_metrics.py. New evaluation rejects positional matching.
"""
from statistics import mean
from .data import validate_records


def token_scores(prediction, reference):
    pred, ref = set(prediction.lower().split()), set(reference.lower().split())
    common = len(pred & ref)
    precision = common / len(pred) if pred else 0.0
    recall = common / len(ref) if ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def align(predictions, references):
    validate_records(predictions)
    validate_records(references)
    def index(records):
        return {(r["record_id"], t["turn_id"]): (r["scan_id"], t)
                for r in records for t in r["turns"]}
    pred, ref = index(predictions), index(references)
    missing, extra = ref.keys() - pred.keys(), pred.keys() - ref.keys()
    if missing or extra:
        raise ValueError(f"ID mismatch: {len(missing)} missing and {len(extra)} unexpected turns. "
                         "Match record_id/turn_id exactly; no rows were scored.")
    result = []
    for key, (scan, truth) in ref.items():
        predicted_scan, candidate = pred[key]
        if scan != predicted_scan:
            raise ValueError(f"Conflicting scan_id for {key!r}; no rows were scored.")
        if truth["question"] != candidate["question"]:
            raise ValueError(f"Conflicting question for {key!r}; restore the same prompt before scoring.")
        result.append((key, candidate["answer"], truth["answer"]))
    return result


def evaluate(predictions, references):
    pairs = align(predictions, references)
    scores = [token_scores(pred, ref) for _, pred, ref in pairs]
    return {"protocol": "answer-only", "unit": "turn", "records": len(references),
            "answers": len(pairs), "excluded": 0,
            "token_precision": mean(s[0] for s in scores),
            "token_recall": mean(s[1] for s in scores),
            "token_f1": mean(s[2] for s in scores)}


def legacy_texts(predictions, references):
    """Explicit positional whole-record extraction for legacy nine-column reports.

    Kept solely for protocol comparison. Dict insertion order and serialized
    formatting affect these strings, as they did in the surviving evaluator.
    """
    if len(predictions) != len(references) or not predictions:
        raise ValueError("Legacy input lists must be nonempty and have equal lengths.")
    def extract(record, reference=False):
        keys = ("text", "answer", "response", "ground_truth" if reference else "output")
        if isinstance(record, dict):
            value = next((record[k] for k in keys if k in record), str(record))
        else:
            value = record
        return "" if value is None else str(value)
    return ([extract(r) for r in predictions], [extract(r, True) for r in references])


def semantic_scores(predictions, references, *, legacy=False, bert_model=None):
    from rouge_score import rouge_scorer
    from bert_score import score
    if not bert_model:
        raise ValueError("Provide --bert-model as an existing local model directory; no implicit download.")
    if legacy:
        pred_texts, ref_texts = legacy_texts(predictions, references)
    else:
        pairs = align(predictions, references)
        pred_texts, ref_texts = [p for _, p, _ in pairs], [r for _, _, r in pairs]
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    bert = score(pred_texts, ref_texts, model_type=bert_model, num_layers=17,
                 idf=False, rescale_with_baseline=False, use_fast_tokenizer=False)
    rows = []
    for i, (pred, ref) in enumerate(zip(pred_texts, ref_texts)):
        rouge = scorer.score(ref, pred)["rougeL"]
        token = token_scores(pred, ref)
        rows.append(dict(zip(("rougeL_precision", "rougeL_recall", "rougeL_f1",
                              "token_precision", "token_recall", "token_f1",
                              "bert_precision", "bert_recall", "bert_f1"),
                             (*rouge, *token, *(float(x[i]) for x in bert)))))
    return {"protocol": "legacy-record" if legacy else "answer-only", "units": len(rows),
            "averages": {k: mean(r[k] for r in rows) for k in rows[0]}, "rows": rows}


def coco_scores(predictions, references):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    pairs = align(predictions, references)
    truth = {i: [r] for i, (_, _, r) in enumerate(pairs)}
    answers = {i: [p] for i, (_, p, _) in enumerate(pairs)}
    result = {"answers": len(pairs), "protocol": "answer-only", "METEOR": "not computed"}
    for name, scorer in [("BLEU_1", Bleu(1)), ("ROUGE_L", Rouge()), ("CIDEr", Cider())]:
        value, _ = scorer.compute_score(truth, answers)
        result[name] = float(value[0] if isinstance(value, list) else value)
    return result
