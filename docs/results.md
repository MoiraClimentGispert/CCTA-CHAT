# Results

I compared my two coronary CT adaptations, FT1 and FT2, with pretrained CT-CHAT 8B and 70B. The evaluation included **627 answers in 252 conversations**, with clinical examples from **two held-out scans**. The answer count reflects multiple questions and dialogue turns, not independent patients.

## Four-model comparison

| Model | BLEU-1 | METEOR¹ | ROUGE-L | CIDEr | Saved LLM-judge mean / 10 |
|---|---:|---:|---:|---:|---:|
| FT1 | 0.280 | 0.170 | 0.297 | 1.009 | 6.435 |
| FT2 | 0.161 | 0.116 | 0.215 | 0.593 | 5.584 |
| CT-CHAT 8B | 0.219 | 0.128 | 0.233 | 0.656 | 5.526 |
| CT-CHAT 70B | 0.234 | 0.139 | 0.252 | 0.756 | 5.611 |

The FT1 scores were recalculated from the predictions used in the thesis. BLEU-1, ROUGE-L and CIDEr were recalculated for all four models and reproduced the thesis table. Judge means summarize the stored scores; the judge was not rerun. ¹METEOR is thesis-reported.

FT1 scored highest on these language metrics. The comparison does not isolate the effect of adapter merging from differences in the other training and inference settings.

## Prompt-alignment sensitivity

Fourteen single-turn short-answer prompts differed between saved predictions and scoring references. Four references remained sufficient; ten involved a changed target or mismatched reference. The following analyses retain separate denominators:

| Subset | Answers | Rule |
|---|---:|---|
| Historical | 627 | Original saved prediction/reference pairing |
| Prompt-identical | 613 | Exclude all fourteen differing prompts |
| Reference-sufficient | 617 | Retain the four differing prompts with sufficient references |
| Local-reference sensitivity | 619 | Also substitute two references available for the same question and scan; exclude the remaining eight |

For the 619-answer analysis, the two replacement references came from the same scan and question elsewhere in the evaluation set. These substitutions test sensitivity to the pairing; they do not change the reported thesis scores.

| Model | 613: BLEU-1 / ROUGE-L / CIDEr | 617: BLEU-1 / ROUGE-L / CIDEr | 619: BLEU-1 / ROUGE-L / CIDEr |
|---|---|---|---|
| FT1 | 0.2796 / 0.3023 / 1.0306 | 0.2796 / 0.3020 / 1.0248 | 0.2797 / 0.3026 / 1.0222 |
| FT2 | 0.1605 / 0.2183 / 0.6053 | 0.1608 / 0.2189 / 0.6024 | 0.1609 / 0.2198 / 0.6013 |
| CT-CHAT 8B | 0.2185 / 0.2359 / 0.6688 | 0.2186 / 0.2366 / 0.6658 | 0.2187 / 0.2373 / 0.6647 |
| CT-CHAT 70B | 0.2334 / 0.2559 / 0.7709 | 0.2336 / 0.2558 / 0.7668 | 0.2339 / 0.2555 / 0.7647 |

**FT1 remained first on recalculated BLEU-1, ROUGE-L and CIDEr in all three subsets.** CIDEr document frequencies were recomputed for each subset. These subsets remove short-answer items, so their task mix differs from the full corpus. The altered reference pairs were not sent back to the judge.

## Answer-only semantic comparison

The historical semantic evaluator serialized whole records, including shared question text and metadata. Answer-only scoring removes that contribution. Here token F1 uses lowercase whitespace-token sets; BERTScore uses RoBERTa-large, layer 17, without IDF or baseline rescaling. The scoring unit matters:

| Model | Whole-record token F1 (252) | Answer-only token F1 (252 records) | Answer-only token F1 (627 turns) | Answer-only BERT F1 (627 turns) |
|---|---:|---:|---:|---:|
| FT1 | 0.5630 | 0.3486 | 0.3387 | 0.9083 |
| FT2 | 0.5354 | 0.2627 | 0.2572 | 0.8902 |
| CT-CHAT 8B | 0.5305 | 0.2872 | 0.2754 | 0.8907 |
| CT-CHAT 70B | 0.5273 | 0.2791 | 0.2939 | 0.8937 |

Record-level answer-only scoring concatenates assistant answers within each conversation before macro-averaging 252 scores. Turn-level scoring macro-averages 627 answers, giving longer dialogues more influence. BERTScore also truncates long inputs at its model limit. These values are language similarity, not clinical accuracy. The public evaluator also reproduced the token-F1 aggregates for all four models on the 613-answer prompt-identical subset.

## Specialist report review

The specialist presentation/quality ratings averaged **4.2/5 for FT1** and **2.4/5 for each comparator**, using five criteria and two scoring-sheet sets. The scored examples used different prompts and scan combinations, so this was not a controlled same-case comparison. Those quality ratings were distinct from judgments of clinical correctness. The reports still contained hallucinations and omissions.

## Qualitative example

In Figure 10 of my thesis (p. 35), FT1 reported minimal narrowing in the left anterior descending artery. It also reported 1–24% narrowing in the left main, where the cardiologist found no lesion.

## Interpretation

I trained on a small cohort concentrated at two disease grades. The results support a comparison of response quality within this dataset, but they do not establish performance across a broader population or intermediate disease grades. The report examples and specialist review also showed that language scores alone were insufficient to judge clinical correctness.

Source: Moira Climent Gispert. *Toward Specialized Med-VLMs: A Vision-Language Approach to Coronary Artery Disease.* Bachelor’s thesis, Universitat Pompeu Fabra, 2025. Tables 1–3 report the quantitative comparison and specialist ratings; Figure 10 (printed p. 35; PDF page 43) shows the qualitative example. The tables above also include subsequent recalculation and prompt-alignment checks on the saved answers. The [quickstart](../README.md#try-the-evaluator) uses a synthetic example and does not reproduce the thesis scores.
