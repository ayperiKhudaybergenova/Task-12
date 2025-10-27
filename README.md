# Task-12

The Abductive Event Reasoning (AER) task is framed as a multiple-choice question answering problem, aiming to evaluate large language models' ability to identify the most plausible direct cause of a real-world event based on textual evidence
for https://sites.google.com/view/semeval2026-task12/


# Handover TODO 

## Goal



## Current Status (Overview)
-reading releated papers.


- https://arxiv.org/pdf/2305.16646

- https://aclanthology.org/2025.acl-long.1269.pdf?utm_source=chatgpt.com

## Known Issues

- 

## Prioritized TODO List

1. Literature review
   
 

1. Control the number of topics per product (high)
   
   - Tune HDBSCAN/UMAP/vectorizer in `fit_bertopic`:
     - Increase `min_cluster_size` and `min_samples` with corpus size to avoid tiny clusters.
     - Optionally reduce topics after fit to a target corridor (e.g., 8–20 per product, configurable).
   - Add explicit per-product target via `.env` (e.g., `TARGET_N_TOPICS_PER_PRODUCT`).

1. Increase topic quality (high)
   
   - Upgrade the embedding model (instead of `all-MiniLM-L6-v2`):
     - Better: `all-mpnet-base-v2`, `multi-qa-mpnet-base-dot-v1`, or E5 family (if available/GPU present). Mind runtime.
   - Improve vectorizer:
     - Domain-specific stopword file; optionally lemmatization or phrases (bi/tri-grams are already on).
     - Set `min_df`/`max_df` dynamically by corpus size to reduce noise.
   - Re-check representation model in BERTopic:
     - Try alternatives to `KeyBERTInspired` (e.g., `MaximalMarginalRelevance`) for more diverse keywords.
   - Make LLM labeling more robust:
     - Add fallbacks and label normalization; perform label deduplication via string similarity/embeddings.

1. Intra-topics (sub-topics) for dense topics (medium)
   
   - Define criteria: if topic document count > X (e.g., ≥ 200) or > Y% of product tickets, then sub-cluster:
     - Extract documents of that topic and run a second BERTopic/HDBSCAN with lower `min_cluster_size`; export separate visuals/CSV.
     - Optionally use hierarchical BERTopic and add `visualize_hierarchy` artifacts.
   - Control thresholds via `.env`: `INTRA_TOPIC_MIN_DOCS`, `INTRA_TOPIC_MIN_SHARE`.

1. Measurement and model selection (medium)
   
   - Compute metrics: Topic Coherence (c_npmi), Topic Diversity, Purity (if labels exist), stability across runs.
   - Small hyperparameter sweeps with metric logging; choose best setting per product.

1. Data quality and cleaning (medium)
   
   - Extend domain-specific cleaning rules (regex/stop-phrases in `processing.py`).
   - Detect weak translations: if repeated very short/identical outputs, consider falling back to original language with multilingual embeddings.

1. Performance/robustness (medium)
   
   - Use GPU for sentence-transformers (if available); adjust batch size.
   - Consider embedding caching and reproducible seeds.
   - Persist runs and parameters (versioning with `run_id` is already present).

## Concrete Work Packages and Hints

A) Merge similar topics

- After fit, create label/keyword embeddings (e.g., mean vector of top words or projection from `topic_model.c_tf_idf_`) and merge pairs above cosine similarity threshold.
- Alternatively, leverage BERTopic `reduce_topics` with `nr_topics='auto'` or a fixed target and/or hierarchical reduction.
- Code location: `app/modeling.py` after `fit_bertopic` and before `apply_labels_and_save_artifacts`.

B) Per-product topic count control

- In `main.py` (per-product path), after each product run, optionally call `reduce_topics` with a product-specific target (new `.env`: `TARGET_N_TOPICS_PER_PRODUCT`).
- In `app/modeling.py:fit_bertopic`, couple `min_cluster_size`/`min_samples` more strongly to corpus size; add caps.

C) Better embeddings/vectorizer

- Replace `SentenceTransformer('all-MiniLM-L6-v2')` with mpnet/e5 variants; benchmark runtime/quality.
- Vectorizer: enable dynamic `min_df`/`max_df` by `len(texts)` (code is present but commented).
- Introduce and load a domain stopword file.

D) Intra-topics

- Add a function like `run_subtopic_model_for_heavy_topics(topic_model, texts, threshold_docs=..., threshold_share=...)`.
- Save results separately: `topic_visual_{product}_{parentTopic}.html`, `topic_labels_{product}_{parentTopic}.csv`.

E) Metrics and selection

- Add `evaluation.py` to compute coherence/diversity (e.g., via gensim or BERTopic utilities) and log to SQLite.
- In `main.py`, add a small sweep runner per product (mind time budget).

## Configuration (.env) – Suggested Additions

- Already present: `TARGET_N_TOPICS`, `MODEL_MAX_TICKETS`, `MODEL_CREATED_AFTER`, `ENABLE_PER_PRODUCT_TOPICS`, `MIN_PRODUCT_GROUP_SIZE`, `REPRESENTATIVE_DOCS`.
- New suggestions:
  - `TARGET_N_TOPICS_PER_PRODUCT=12`
  - `HDBSCAN_MIN_CLUSTER_PCT=0.002` (lower bound for `min_cluster_size` w.r.t. corpus size)
  - `VECTOR_MIN_DF=2`, `VECTOR_MAX_DF=0.85`
  - `TOPIC_MERGE_MIN_SIM=0.85`
  - `INTRA_TOPIC_MIN_DOCS=200`, `INTRA_TOPIC_MIN_SHARE=0.25`

## How to Run (Short)

- Global: `python main.py`
- Per product: `python main.py --per-product` or set `.env: ENABLE_PER_PRODUCT_TOPICS=true`
- Artifacts: `output/` includes visuals (`topic_visual_*.html`, `topic_barchart_*.html`), labels (`topic_labels_*.csv`) and `topic_results.db`.

## Definition of Done

- Significantly fewer duplicate/near topics (qualitative + similarity score decreases).
- Reduced but sensible topic count per product (target corridor reached).
- Improved topic quality (higher coherence/acceptable diversity).
- Sub-topic analyses and visuals exist for dense topics.

## Code Pointers

- `app/modeling.py`
  - `fit_bertopic`: adjust hyperparameters (UMAP/HDBSCAN/vectorizer/embeddings).
  - `generate_topic_labels`: LLM labeling; add label normalization/deduplication.
  - Post-fit reduction/merge: new block before `apply_labels_and_save_artifacts`.
- `main.py`
  - Per-product flow in `_run_per_product`; apply product-specific target topic counts/thresholds.
- `app/processing.py`
  - Extend cleaning/stop-phrases; adjust product grouping (`PRODUCT_SUBSTRING_MAPPING`).
