"""ID-preserving preparation derived from the typed CT-CHAT thesis unifier.

Release reconstruction: inputs are already curated QA, not clinical source records.
The same record/scan/turn identity is used in preparation, inference and evaluation.
"""
import hashlib
import json
from pathlib import Path

TASKS = {"short_answer", "long_answer", "multiple_choice", "report_generation"}


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_records(records, *, answers=True):
    """Require explicit identities; never align on question wording alone."""
    if not isinstance(records, list) or not records:
        raise ValueError("Input must be a nonempty JSON list of records.")
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each record must be an object.")
        for name in ("record_id", "scan_id"):
            if not isinstance(record.get(name), str) or not record[name].strip():
                raise ValueError(f"Every record needs a nonempty string {name}.")
        rid = record["record_id"]
        if rid in seen:
            raise ValueError(f"Duplicate record_id {rid!r}; use distinct record IDs.")
        seen.add(rid)
        turns = record.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"Record {rid!r} needs a nonempty turns list.")
        ids = set()
        for turn in turns:
            if not isinstance(turn, dict):
                raise ValueError(f"Record {rid!r}: each turn must be an object.")
            tid = turn.get("turn_id")
            if not isinstance(tid, str) or not tid.strip():
                raise ValueError(f"Record {rid!r}: every turn needs a string turn_id.")
            if tid in ids:
                raise ValueError(f"Duplicate turn_id {tid!r} in record {rid!r}.")
            ids.add(tid)
            for name in (("question", "answer") if answers else ("question",)):
                if not isinstance(turn.get(name), str) or not turn[name].strip():
                    raise ValueError(f"Record {rid!r}, turn {tid!r}: {name} must be nonempty text.")
    return records


def prepare(records):
    """Turn curated QA into CT-CHAT conversations; literature stays image-free."""
    validate_records(records)
    result = []
    splits = {}
    for record in records:
        task = record.get("task", "short_answer")
        if task not in TASKS:
            raise ValueError(f"Unsupported task {task!r}; choose {sorted(TASKS)}.")
        image = record.get("image")
        if image is not None and (not isinstance(image, str) or not image.endswith(".npz")):
            raise ValueError("image must be a .npz filename or null.")
        if record.get("source") == "literature":
            if image:
                raise ValueError("Literature records must be text-only (image=null).")
            if not record.get("article_id"):
                raise ValueError("Literature records need article_id to audit split overlap.")
        if not isinstance(record.get("split"), str) or not record["split"].strip():
            raise ValueError("Prepared records need an explicit split.")
        identity = ("article", record["article_id"]) if record.get("source") == "literature" else \
                   ("scan", record["scan_id"]) if image else None
        if identity:
            prior = splits.setdefault(identity, record["split"])
            if prior != record["split"]:
                raise ValueError(f"The same {identity[0]} appears in different splits; separate independent sources first.")
        conversations = []
        for i, turn in enumerate(record["turns"]):
            if any(marker in turn["question"] for marker in ("<image>", "<provided>")):
                raise ValueError("Provide unmarked questions; prepare adds image/provided markers.")
            prefix = "<image>\n<provided>\n" if image and i == 0 else ""
            question = f"{prefix}<{task}> {turn['question']}"
            conversations.extend([{"from": "human", "value": question, "turn_id": turn["turn_id"]},
                                  {"from": "gpt", "value": turn["answer"], "turn_id": turn["turn_id"]}])
        result.append({k: v for k, v in record.items() if k != "turns"} |
                      {"image": image, "type": task, "conversations": conversations})
    return result


def article_prompt(title, body):
    """Historical enhanced-generator input length: 4,000 characters, not tokens."""
    return ("Create five educational question-answer pairs about coronary CT angiography. "
            "Use only the supplied article text; return JSON with question and answer fields.\n\n"
            f"{title}\n\n{body[:4000]}")


def retrieve_articles(query, limit, split):
    """Retrieve PMC articles with IDs; extraction includes nested XML body text."""
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen
    from xml.etree import ElementTree as ET
    if not query.strip() or not 1 <= limit <= 100:
        raise ValueError("Use a nonempty search query and a limit between 1 and 100.")
    def fetch(endpoint, params):
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/" + endpoint + "?" + urlencode(params)
        with urlopen(Request(url, headers={"User-Agent": "CCTA-CHAT/0.1 (research preparation)"}), timeout=60) as response:
            return ET.fromstring(response.read())
    search = fetch("esearch.fcgi", {"db": "pmc", "term": query, "retmax": limit, "retmode": "xml"})
    identifiers = [node.text for node in search.findall("./IdList/Id")]
    if not identifiers:
        raise ValueError("No PMC articles matched the supplied query.")
    articles = fetch("efetch.fcgi", {"db": "pmc", "id": ",".join(identifiers), "retmode": "xml"})
    result = []
    for article in articles.findall(".//article"):
        def text(path):
            node = article.find(path)
            return " ".join(node.itertext()).strip() if node is not None else ""
        identity = text(".//article-id[@pub-id-type='pmc']")
        title, body = text(".//article-title"), text("./body")
        if identity and title and body:
            result.append({"article_id": identity, "title": title, "body": body, "split": split,
                           "source": "PMC", "license_text": text(".//permissions")})
    if not result:
        raise ValueError("Retrieved articles contained no usable full-text bodies.")
    return result
