# What CCTA-CHAT changes

The starting point is [CT-CHAT, commit `33c9b973bf1babb3b79b5ffd8ae45948e758a3f2`](https://github.com/ibrahimethemhamamci/CT-CHAT/tree/33c9b973bf1babb3b79b5ffd8ae45948e758a3f2), developed by Hamamci and colleagues. CT-CHAT contributes the pretrained chest-CT workflow and multimodal model; CT-CLIP contributes its CT-ViT visual encoder. CT-RATE is their chest-CT dataset and a hosting location for model artifacts. The CCTA thesis cohort is a separate study.

I used CT-CHAT’s model architecture as the starting point and adapted the data, prompts and fine-tuning workflow for coronary CT. This repository packages the preparation, training, inference and evaluation commands. Shared model code stays in a pinned upstream checkout, with changes supplied by the [integration patch](../src/ccta_chat/integration/ctchat.patch).

## Component change map

| What CT-CHAT supplied | CCTA-CHAT path | What changed and why | When |
|---|---|---|---|
| Chest-CT tasks and supervision | `data.py`, `prepare` | Coronary questions and report tasks, image/task markers and text-only literature | Thesis adaptation; IDs, splits and validation added to the public code |
| Literature preparation scripts | `data.py`, `generation.py` | PMC retrieval and five QA pairs from each title and first 4,000 body characters; article IDs and complete XML text extraction | Thesis workflow; parsing and provenance improvements in the public code |
| `llava/serve/encode_script.py` | `encoding.py` | CCTA spacing/window/crop; header spacing and cache fingerprints | Thesis CCTA profile; explicit preprocessing and reuse checks in the public code |
| `llava/conversation.py`, `llava/constants.py` | [Integration patch](../src/ccta_chat/integration/ctchat.patch) | CCTA/CAD-RADS prompt and consistent short-answer marker | Thesis adaptation; tokenizer compatibility checks in the public code |
| `llava/train/train.py`, entry point and trainer | `training.py`, `model.py`, `configs/` | FT1 fresh adapter or FT2 adapter continuation, with a trainable projector; explicit configuration, masking and mixed batches | Thesis strategies; loading checks and run metadata in the public code |
| `llava/model/builder.py` and merge helpers | `model.py`, `merge-base` | Base/adapter/projector loading and merged-base export, with key/shape checks | Thesis integration; consolidated commands in the public code |
| `llava/serve/ctchat_validation_llama.py` | `model.py`, `infer` | Image and text-only answers, attention masks, deterministic generation and stable record/turn IDs | Thesis validation workflow; configurable inputs and outputs in the public code |
| `llava/model/llava_arch.py`, package imports and `llava/mm_utils.py` | Integration patch → the same upstream paths | Keep cached-feature insertion and token construction; remove inactive tower dependencies and correct LLaMA imports | Public integration cleanup |
| COCO/semantic evaluation and LLM judging scripts | `evaluation.py`, `generation.py` | ID-aligned answer scoring, an explicit whole-record legacy mode and saved judge triplets; no heuristic GREEN score | Thesis evaluation; alignment and reproducibility improvements in the public code |
| **Unchanged:** `llava/model/language_model/llava_llama.py`, `multimodal_projector/builder.py`, `coca_attentional_pooler.py` | Separate pinned checkout; hashes in `integration/backend.json` | CT-CHAT’s LLaMA wrapper, attention pool and MLP projector are used as provided | **Upstream CT-CHAT/LLaVA code** |

Package paths above are relative to `src/ccta_chat/`. The `prepare` command starts with curated English QA. [Methods](methods.md) explains the command inputs; [Results](results.md) presents the thesis comparison.

## Pinned model integration

`ccta-chat integrate` checks the exact upstream revision before applying the patch. The public commands verify fingerprints for the changed modules and unchanged transitive model dependencies. The commands import that checkout explicitly, so a different installed `ctchat-llava` package cannot silently replace it.

The offline encoder resolves `CTViT` from [CT-CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP/tree/a2a155c601987820433c01db69b64d701d3d229d), release dependency revision `a2a155c601987820433c01db69b64d701d3d229d`, using its `transformer_maskgit` package. This dependency is kept separate from the language-model backend.

## Credits and source figures

My work builds on CT-CHAT and CT-CLIP. Please credit their authors and cite *Developing Generalist Foundation Models from a Multimodal Dataset for 3D Computed Tomography* ([paper](https://arxiv.org/abs/2403.17834)). The LLaVA-derived files retain their original author notices. The thesis adaptation is by **Moira Climent Gispert**.

| Displayed figure | Thesis source (2025) | Edits |
|---|---|---|
| README workflow | *Presentation Methodology pipeline.png* | Correct “Data Aquisition” to “Data acquisition”; keep the rest of the original figure |
| Methods: FT1 | *Training approach 1.png* | None |
| Methods: FT2 | *Training approach 2.png* | None |

The training figures are stacked and link to their full-resolution images. Each caption credits the original or adapted thesis figure.

See [third-party notices](../THIRD_PARTY_NOTICES.md), [CC BY-NC-SA 4.0](../LICENSE) and the retained [Apache 2.0 text](../APACHE-2.0.txt). The CT-CHAT publisher's license statement is preserved by reference to its [pinned README](https://github.com/ibrahimethemhamamci/CT-CHAT/blob/33c9b973bf1babb3b79b5ffd8ae45948e758a3f2/README.md#license). Model and dataset artifacts have their own terms and are not included in the code repository.
