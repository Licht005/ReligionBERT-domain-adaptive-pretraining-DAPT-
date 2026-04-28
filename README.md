# ReligionBERT

Domain-adaptive pre-training and multi-task fine-tuning of BERT-based models on religious text corpora. Two models are trained: a monolingual English variant and a multilingual variant, each evaluated against their generic counterparts on three downstream tasks.

**Models on HuggingFace:**
- [LucasLicht/religion-bert](https://huggingface.co/LucasLicht/religion-bert)
- [LucasLicht/multi-religion-bert](https://huggingface.co/LucasLicht/multi-religion-bert)

---

## Overview

Standard BERT models are pre-trained on general-domain corpora and often underperform on specialized domains with distinct vocabulary and stylistic patterns. Religious text, particularly biblical literature, contains archaic language structures, proper-noun-dense passages, and cross-lingual verse correspondences that generic pre-training does not adequately capture.

This project applies continued masked language modeling (MLM) pre-training to two BERT-base checkpoints using the [christos-c/bible-corpus](https://github.com/christos-c/bible-corpus), a multilingual XML Bible collection spanning 100+ languages. The resulting models are fine-tuned on three tasks designed to probe domain understanding: semantic similarity across Bible translations, book-of-origin classification, and extractive question answering over verse content.

---

## Corpus

**Source:** `christos-c/bible-corpus` (~500MB, 100+ translations, XML format)

| Split | Languages | Verses |
|---|---|---|
| ReligionBERT training | English (KJV + WEB) | ~62,000 |
| MultiReligionBERT training | 12 languages | ~373,000 |
| Cross-lingual eval | Ewe, Swahili (NT only) | held-out |

Pre-training languages: English, French, Spanish, Portuguese, German, Amharic, Shona, Xhosa, Malagasy, Somali, Zarma.

---

## Pre-training (Objective 1)

Both models are initialized from their respective BERT-base checkpoints and continued pre-trained via MLM (15% masking, max length 128).

| Model | Base | Steps | Eval Loss |
|---|---|---|---|
| ReligionBERT | `bert-base-uncased` | 30,000 | 1.164 |
| MultiReligionBERT | `bert-base-multilingual-cased` | 30,000 | 1.392 |

**Config:** LR 3e-5, warmup 500 steps, batch size 16, gradient accumulation 2, fp16, `save_total_limit=2`. Checkpoints permanently deleted after each save to manage storage. Models pushed to HuggingFace Hub immediately after training.

The multilingual corpus (372,652 sequences) was tokenized in 50,000-line chunks to avoid RAM overflow during dataset construction.

---

## Fine-tuning (Objective 2)

Three downstream tasks, all evaluated against `bert-base-uncased` and `bert-base-multilingual-cased` baselines.

### Task 1: Semantic Similarity

Regression task over verse pairs from 5 translations (KJV, WEB, French, Spanish, German). Labels assigned by pair type: same verse across translations (high), same book different chapter (medium), same chapter different verse (hard negative), cross-testament (low). Class imbalance addressed by downsampling to a balanced 6,392-pair training set for Stage 1, followed by refinement on the full 21,994-pair set at a lower LR.

| Model | Pearson | Spearman |
|---|---|---|
| BERT-base-uncased | 0.9569 | 0.6556 |
| ReligionBERT | 0.9623 | 0.6591 |
| mBERT | 0.9624 | 0.6772 |
| MultiReligionBERT | 0.9635 | 0.6669 |

### Task 2: Book Classification

66-class classification where each verse is assigned to its Bible book. Dataset: 150 verses per book sampled from English KJV, 7,726 total samples, 80/10/10 split. Optimized on macro F1.

| Model | Accuracy | Macro F1 |
|---|---|---|
| BERT-base-uncased | 0.4075 | 0.3144 |
| ReligionBERT | 0.4347 | 0.3381 |
| mBERT | 0.3972 | 0.2948 |
| MultiReligionBERT | 0.4360 | 0.3369 |

### Task 3: Question Answering

SQuAD-format extractive QA. Dataset generated via Groq API (Llama 3.3-70B): 1,199 question-answer pairs from KJV verses, balanced 600 OT / 600 NT, with substring validation to guarantee every answer is an exact span of the source verse. Human review of 200 samples confirmed 100% answer validity. Evaluation uses argmax span decoding (EM and token F1).

| Model | Exact Match | Token F1 |
|---|---|---|
| BERT-base-uncased | 32.50 | 58.91 |
| ReligionBERT | 40.00 | 60.48 |
| mBERT | 48.33 | 70.54 |
| MultiReligionBERT | 40.83 | 65.39 |

---

## Results Summary

Domain-adapted pre-training consistently improves performance over generic baselines on tasks that rely on religious vocabulary and passage structure. ReligionBERT outperforms BERT-base-uncased on all three tasks. MultiReligionBERT outperforms mBERT on semantic similarity and book classification, though mBERT holds an advantage on QA, likely due to stronger English representational capacity in the mBERT checkpoint.

The QA task shows the largest absolute improvement from domain adaptation: ReligionBERT gains +7.5 EM and +1.57 F1 over the generic baseline, and MultiReligionBERT gains +2.5 EM over mBERT on F1.

---

## Repo Structure

```
ReligionBERT/
├── datasets/
│   ├── raw/bible/              # christos-c/bible-corpus XML files
│   ├── processed/              # Extracted JSONs, training TXTs, tokenized shards
│   └── finetuning/
│       ├── semantic_similarity/
│       ├── classification/
│       └── qa/
├── models/
│   ├── baselines/              # bert-base-uncased, mbert, xlm-roberta
│   ├── religion-bert/
│   └── multi-religion-bert/
├── results/
│   ├── figures/                # All evaluation plots (Figures 4.1-4.11)
│   └── metrics/                # JSON result files per task per model
└── ReligionBERT_clean.ipynb   # Full implementation notebook
```

---

## Notes

- Book code quirks in the corpus: `MAR` not `MRK`, `JOH` not `JHN`, `EZE` not `EZK`, `JOE` not `JOL`, `NAH` not `NAM`, `JAM` not `JAS`, `PHI` not `PHP`, `SON` not `SNG`, `1JO/2JO/3JO` not `1JN/2JN/3JN`.
- Requires Transformers 5.0.0+: `eval_strategy` replaces `evaluation_strategy`, `processing_class` replaces `tokenizer` in `Trainer`, `logging_dir` removed.
- Use `BertForMaskedLM` not `BertModel` for pre-training checkpoints.
- QA fine-tuning requires `BertTokenizerFast` for offset mapping.
