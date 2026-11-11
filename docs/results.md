# Results

The thesis compares CAD CT-CHAT FT1 and FT2 with pretrained CT-CHAT 8B and 70B. The historical evaluation contains **627 answers in 252 conversations**; clinical examples cover **two held-out scans**. The answers are not independent patient observations.

## Historical comparison

| Model | BLEU-1 | METEOR¹ | ROUGE-L | CIDEr | Saved LLM-judge mean / 10 |
|---|---:|---:|---:|---:|---:|
| FT1 | 0.280 | 0.170 | 0.297 | 1.009 | 6.435 |
| FT2 | 0.161 | 0.116 | 0.215 | 0.593 | 5.584 |
| CT-CHAT 8B | 0.219 | 0.128 | 0.233 | 0.656 | 5.526 |
| CT-CHAT 70B | 0.234 | 0.139 | 0.252 | 0.756 | 5.611 |

BLEU-1, ROUGE-L and CIDEr were recalculated from saved predictions and references and reproduce the thesis table. Judge means aggregate stored scores; the judge was not rerun. ¹METEOR is thesis-reported. The original FT1 prediction variant is the historical result anchor; a separate later prediction variant is not substituted into this table.

FT1 leads on these language metrics. This within-study comparison does not isolate a causal effect of adapter merging from the other training and inference settings.

## Prompt-alignment sensitivity

Fourteen single-turn short-answer prompts differed between saved predictions and scoring references. Four references remained sufficient; ten involved a changed target or mismatched reference. The following analyses retain separate denominators:

| Subset | Answers | Rule |
|---|---:|---|
| Historical | 627 | Original saved prediction/reference pairing |
| Prompt-identical | 613 | Exclude all fourteen differing prompts |
| Reference-sufficient | 617 | Retain the four differing prompts with sufficient references |
| Local-reference sensitivity | 619 | Also substitute two references available for the same question and scan; exclude the remaining eight |

The substitutions are a sensitivity analysis, not a recovered earlier reference version. No row-level clinical comparisons are published.

| Model | 613: BLEU-1 / ROUGE-L / CIDEr | 617: BLEU-1 / ROUGE-L / CIDEr | 619: BLEU-1 / ROUGE-L / CIDEr |
|---|---|---|---|
| FT1 | 0.2796 / 0.3023 / 1.0306 | 0.2796 / 0.3020 / 1.0248 | 0.2797 / 0.3026 / 1.0222 |
| FT2 | 0.1605 / 0.2183 / 0.6053 | 0.1608 / 0.2189 / 0.6024 | 0.1609 / 0.2198 / 0.6013 |
| CT-CHAT 8B | 0.2185 / 0.2359 / 0.6688 | 0.2186 / 0.2366 / 0.6658 | 0.2187 / 0.2373 / 0.6647 |
| CT-CHAT 70B | 0.2334 / 0.2559 / 0.7709 | 0.2336 / 0.2558 / 0.7668 | 0.2339 / 0.2555 / 0.7647 |

**FT1 remains first on recalculated BLEU-1, ROUGE-L and CIDEr in all three subsets.** CIDEr document frequencies were recomputed for each subset. These subsets remove short-answer items, so their task mix differs from the full corpus. Historical judge means for modified reference pairs are not presented as newly judged scores.

## Answer-only semantic comparison

The historical semantic evaluator serialized whole records, including shared question text and metadata. Answer-only scoring removes that contribution. Here token F1 uses lowercase whitespace-token sets; BERTScore uses RoBERTa-large, layer 17, without IDF or baseline rescaling. The scoring unit matters:

| Model | Whole-record token F1 (252) | Answer-only token F1 (252 records) | Answer-only token F1 (627 turns) | Answer-only BERT F1 (627 turns) |
|---|---:|---:|---:|---:|
| FT1 | 0.5630 | 0.3486 | 0.3387 | 0.9083 |
| FT2 | 0.5354 | 0.2627 | 0.2572 | 0.8902 |
| CT-CHAT 8B | 0.5305 | 0.2872 | 0.2754 | 0.8907 |
| CT-CHAT 70B | 0.5273 | 0.2791 | 0.2939 | 0.8937 |

Record-level answer-only scoring concatenates assistant answers within each conversation before macro-averaging 252 scores. Turn-level scoring macro-averages 627 answers, giving longer dialogues more influence. BERTScore also truncates long inputs at its model limit. These values are language similarity, not clinical accuracy. The maintained evaluator independently matches all four known token-F1 aggregates on the 613-answer prompt-identical subset.

## Specialist report review

Recovered presentation/quality means are **4.2/5 for FT1** and **2.4/5 for each comparator**, using five criteria and two scoring-sheet sets. The scored examples used different prompts and scan combinations; this was not a controlled same-case comparison. The quality ratings are distinct from correctness judgments. Hallucinations and omissions remained in the generated reports.

## Interpretation

The cohort covers a small set of scans and training disease grades. Repeated conversational answers do not expand the number of independent patients. The study supports a comparison of response quality within this dataset; it does not establish broad generalization, intermediate-grade performance or clinical readiness. Full-scale re-evaluation of a model release is a separate experiment from reproducing aggregate scores from saved responses.

Source: Moira Climent's bachelor’s thesis, *Toward Specialized Med-VLMs: A Vision-Language Approach to Coronary Artery Disease* (2025), quantitative evaluation and specialist review, with subsequent recalculation and prompt-alignment checks on the saved evaluation artifacts. All tables here contain aggregate results only. The [quickstart](../README.md#run-the-evaluation-example) uses an independently authored synthetic fixture and does not reproduce these numbers.
