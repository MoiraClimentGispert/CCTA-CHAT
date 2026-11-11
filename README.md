# CCTA-CHAT

**Moira Climent Gispert · Bachelor's thesis · Universitat Pompeu Fabra · 2025**

For my bachelor’s thesis at Universitat Pompeu Fabra, I wanted to see how well a model trained on chest CT could answer questions about individual coronary arteries. I adapted [CT-CHAT](docs/upstream.md) for coronary CT angiography using specialist reports and structured annotations from Hospital de la Santa Creu i Sant Pau. I used eight scans for training and two held-out scans for evaluation.

[![Four-stage thesis workflow: acquisition of CCTA volumes, reports and structured annotations; dataset curation; CT-CHAT fine-tuning; and evaluation comparing a model answer with the reference answer.](docs/figures/workflow.png)](docs/figures/workflow.png)

*Adapted from Moira Climent Gispert’s thesis figure (2025).* [Open the full-size workflow.](docs/figures/workflow.png)

I compared two ways to fine-tune the model. **FT1** merges the original CT-CHAT adapter into the base model and trains a new adapter for coronary CT. **FT2** continues training an existing CT-CHAT adapter. Both strategies also train the projector that connects the scan features to the language model.

**FT1 scored highest on BLEU-1, ROUGE-L and CIDEr across 627 answers in 252 conversations.** Reviewing the generated reports showed why those scores tell only part of the story. In one example, the model identified minimal narrowing in the left anterior descending artery but also reported a left-main lesion where the cardiologist found none. I discuss this finding from Figure 10 of my thesis in the [case discussion](docs/results.md#qualitative-example). The [results page](docs/results.md) compares all four models.

## Try the evaluator

You can try the evaluator on four synthetic answers using Python and a CPU. Use Python 3.10 or later. The clinical data and evaluated CAD weights are unavailable in this release.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
ccta-chat evaluate --references examples/references.json --predictions examples/predictions.json
```

Expected output:

```text
CCTA-CHAT | answer-only evaluation
Records: 3 | Answers: 4 | Excluded: 0
Token precision: 0.875000
Token recall:    0.750000
Token F1:        0.791667
Aggregation: macro-average over turns (lowercase whitespace-token sets)
Synthetic fixture: illustrative; not thesis results.
```

The evaluator matches each prediction to its question by ID and scores the answer text.

Read [Methods](docs/methods.md) for data preparation, scan features, FT1/FT2 and model commands; [Results](docs/results.md) for the four-model comparison and sensitivity analyses; and [the upstream change map](docs/upstream.md) for what CT-CHAT supplied and what I changed for CCTA.

## Tests

```bash
python -m unittest discover -s tests -v
```

The base install has no runtime dependencies. Tests cover evaluation alignment and data preparation, with optional synthetic model checks described in [Methods](docs/methods.md#tests). Python 3.11 is recommended for the optional model dependencies in `pyproject.toml`.

## Credits

My work builds on CT-CHAT and CT-CLIP by Hamamci and colleagues. The shared model code is pinned to CT-CHAT commit [`33c9b97`](https://github.com/ibrahimethemhamamci/CT-CHAT/tree/33c9b973bf1babb3b79b5ffd8ae45948e758a3f2). See the [upstream credits](docs/upstream.md#credits-and-source-figures) and [third-party notices](THIRD_PARTY_NOTICES.md).

Moira Climent Gispert. *Toward Specialized Med-VLMs: A Vision-Language Approach to Coronary Artery Disease.* Bachelor’s thesis, Universitat Pompeu Fabra, 2025.
