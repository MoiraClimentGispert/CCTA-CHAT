# What CCTA-CHAT changes

The starting point is [CT-CHAT, commit `33c9b973bf1babb3b79b5ffd8ae45948e758a3f2`](https://github.com/ibrahimethemhamamci/CT-CHAT/tree/33c9b973bf1babb3b79b5ffd8ae45948e758a3f2), developed by Hamamci and colleagues. CT-CHAT contributes the pretrained chest-CT workflow and multimodal model; CT-CLIP contributes its CT-ViT visual encoder. CT-RATE is their chest-CT dataset and a hosting location for model artifacts. The CCTA thesis cohort is a separate study.

This standalone repository contains the CCTA adaptation and maintained interfaces. Shared model machinery stays in a pinned upstream checkout. Each modification is represented once: either in the small package or in the packaged [integration patch](../src/ccta_chat/integration/ctchat.patch). No upstream history or duplicate model source tree is imported.

## Component change map

| Upstream path or behavior | Public path | Change and purpose | Origin |
|---|---|---|---|
| Chest-CT tasks and supervision | `src/ccta_chat/data.py` | CCTA task types and marked image conversations; literature remains text-only | Thesis adaptation; IDs, split/article provenance and validation maintained for release |
| Literature preparation additions | `data.py`, `generation.py` | Explicit PMC query, complete nested XML text, title + 4,000 body characters and five QA pairs; local-model JSON generation | Thesis workflow; extraction, schema and provenance cleanup for release |
| `llava/serve/encode_script.py` chest preprocessing | `src/ccta_chat/encoding.py` | CCTA 0.35/0.35/0.30-mm spacing, −100…800 HU, [0,1] normalization and 480×480×240 crop | Thesis CCTA variant; header spacing, native-axis documentation and fingerprints added for release |
| Batch loop and skip-existing variant | `encoding.py`, `encode --reuse` | Reuse only when input/source/encoder/preprocessing/output fingerprints agree | Release maintenance of batch reuse behavior |
| `llava/conversation.py` chest system prompt | `integration/ctchat.patch` → same upstream path | CCTA and CAD-RADS conversation template | Thesis adaptation |
| `llava/constants.py` short-answer marker | Same patch → same upstream path | Consistent `<short_answer>` spelling | CCTA source variant retained in release; tokenizer compatibility is checked |
| `llava/train/train.py`, entry point and trainer | `src/ccta_chat/training.py`, `model.py`, `configs/ft1.json`, `configs/ft2.json` | Compact cached-feature trainer; explicit fresh/continued LoRA initialization, assistant-span masking, LoRA + projector trainability, cosine/warmup and explicit resume | Thesis strategies; initialization checks, mixed batches, shared limits and run manifests maintained for release |
| `llava/model/builder.py` and merge helpers | `src/ccta_chat/model.py`, `merge-base` | Explicit local base/adapter/projector loading and merged-base export; normalize projector prefixes, require full key/shape coverage, verify FT2 tensor restoration | Thesis integration approach; consolidated loader and artifact checks for release |
| `llava/serve/ctchat_validation_llama.py` | `model.py`, `infer` | Image and text-only inference, deterministic generation, attention masks, stable record/turn IDs and configurable outputs | Thesis validation workflow; maintained IO and provenance |
| `llava/model/llava_arch.py` cached features | Patch → same upstream path | Remove inactive live-tower import; retain cached-feature insertion and projector path | Release import cleanup; no live-tower tuning is exposed |
| `llava/__init__.py`, `llava/model/__init__.py` | Patch → same upstream paths | Correct package entry point; import only supported LLaMA backend | Release import cleanup |
| `llava/mm_utils.py` token construction | Patch → same upstream path | Retain selected `as_tensor` construction for token IDs | Retained source variant; integer IDs are not a gradient fix |
| Local COCO metric additions | `src/ccta_chat/evaluation.py`, `coco` | BLEU-1/ROUGE-L/CIDEr definitions, with ID alignment and explicit denominator | Thesis evaluation; maintained alignment and optional dependencies |
| Nine-column semantic evaluator | `evaluation.py`, `semantic` | Explicit whole-record legacy profile plus revised answer-only profile; omit heuristic GREEN | Historical protocol; maintained extraction/alignment and local-model control |
| LLM judge and aggregate analysis | `generation.py`, `judge`; [results](results.md) | Preserve full triplets for new judging and report stored historical score means separately | Thesis evaluation; maintained reproducibility |
| **Unchanged:** `llava/model/language_model/llava_llama.py`, `multimodal_projector/builder.py`, `coca_attentional_pooler.py` | Separate pinned checkout; hashes in `integration/backend.json` | CT-CHAT's LLaMA wrapper, attention pool and MLP projector are used as provided | **Upstream CT-CHAT/LLaVA machinery** |

The archive's broader clinical preprocessing and prompt experiments informed the thesis. The public preparation interface begins at curated English QA and does not include hospital files, translations of private records or training/prediction outputs. The maintained modules consolidate the selected operations instead of publishing every experimental script. The [methods](methods.md) document the input contracts and the release configurations; the [results](results.md) describe the historical evaluation separately.

## Integration and attribution

`ccta-chat integrate` checks the exact upstream revision before applying the patch. The public commands verify fingerprints for the changed modules and unchanged transitive model dependencies. The package does not depend on an ambient `ctchat-llava` installation.

The offline encoder resolves `CTViT` from [CT-CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP/tree/a2a155c601987820433c01db69b64d701d3d229d), release dependency revision `a2a155c601987820433c01db69b64d701d3d229d`, using its `transformer_maskgit` package. This dependency is kept separate from the language-model backend.

Credit the upstream CT-CHAT/CT-CLIP authors and cite *Developing Generalist Foundation Models from a Multimodal Dataset for 3D Computed Tomography* ([paper](https://arxiv.org/abs/2403.17834)). The LLaVA-derived files retain their original author notices. The thesis adaptation is by **Moira Climent**. New generic schematics were recreated for this release from the thesis workflow and intended training diagrams; no scan slices or patient text were copied.

See [third-party notices](../THIRD_PARTY_NOTICES.md), [CC BY-NC-SA 4.0](../LICENSE) and the retained [Apache 2.0 text](../APACHE-2.0.txt). The CT-CHAT publisher's license statement is preserved by reference to its [pinned README](https://github.com/ibrahimethemhamamci/CT-CHAT/blob/33c9b973bf1babb3b79b5ffd8ae45948e758a3f2/README.md#license). Model and dataset artifacts have their own terms and are not included in the code repository.
