AgentRisk — Subtask 1: Risk Severity Estimation
Status: Baseline research phase archived; final competition system not yet selected.
Competition: SemEval 2027 AgentRisk, Subtask 1
Task: Predict the severity (an integer from 1–5) of an AI agent's proposed action given its interaction context.
1. Project overview
This project investigates how well conventional text features and pretrained sentence embeddings predict the risk severity of proposed AI-agent actions. We started with simple classification baselines, diagnosed input truncation in MiniLM, experimented with action-aware and chunked-context embeddings, and introduced scenario-grouped cross-validation to make comparisons more rigorous.

Research direction: Can predicted harm domains and risk drivers (from Subtask 2) provide useful auxiliary information to improve Subtask 1 severity estimation?
2. Dataset and evaluation
- Source: Official AgentRisk dataset on Hugging Face.
- Training split: 568 examples.
- Development split: 143 examples.
- Target: risk_severity (levels 1–5).
- Main text fields: formatted_interaction and proposed_action.
- Main metric: Mean Absolute Error (MAE); lower is better. The theoretical range for predictions and labels restricted to 1–5 is 0–4.
- Secondary metric: Exact-match accuracy.
Scenario-grouped cross-validation
We used StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42) on the 568 training examples only. Group IDs were matched by example_id from the official split_manifest.jsonl file. The three validation folds contained 193, 180, and 195 examples respectively, with zero overlapping scenario groups between training and validation in every fold. These folds are not perfectly class-balanced: there were only 7 risk-level-5 examples across the training set, and the third validation fold had none.

For model comparisons, we use overall out-of-fold (OOF) MAE across all 568 training examples rather than treating the separate 143-example development set as a tuning set.
3. Experimental results
A. Grouped cross-validation (568 training examples)
Model	OOF MAE ↓	OOF accuracy
TF-IDF + Logistic Regression	0.4877	67.61%
Action-only MiniLM embeddings + Logistic Regression	0.6232	58.98%
Chunked context + action MiniLM embeddings + Logistic Regression	0.5546	61.27%


Current grouped-CV baseline: TF-IDF + Logistic Regression.
B. Official development set (143 examples)
Model	Dev MAE ↓	Dev accuracy
TF-IDF + Logistic Regression	0.4825	67.83%
Original MiniLM embeddings + Logistic Regression	1.1678	39.16%
Action-only MiniLM embeddings + Logistic Regression	0.6503	57.34%
Original TF-IDF + MiniLM + LightGBM hybrid	0.3147	72.03%


Important: The hybrid's 0.3147 MAE was measured on the development set, not through the above grouped-CV procedure. It must be evaluated on the same grouped folds before calling it the best-validated model. The more recently verified numbers above supersede older, inconsistent development-set scores.
4. Methods and experiments
TF-IDF + Logistic Regression baseline
- Represents text using weighted unigram and bigram features.
- Grouped-CV configuration: TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True); LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42).
- TF-IDF is fit inside each training fold, then used to transform that fold's validation examples, preventing preprocessing leakage.
Original MiniLM baseline and truncation investigation
- Encoder: sentence-transformers/all-MiniLM-L6-v2 (384-dimensional embeddings; 256-token sequence limit in this setup).
- Diagnostic token counts for the original combined text inputs:
  - Training: average 1,076.4 tokens, maximum 3,853; 568/568 exceeded 256 tokens.
  - Development: average 1,098.0 tokens, maximum 2,865; 143/143 exceeded 256 tokens.
- This means the original encoder discarded text beyond the model limit. Because the proposed action was appended late in the input representation, it could be lost through truncation.
- This is evidence of a representation problem, not proof that truncation alone caused the model's poor performance.
Action-only MiniLM
- Encodes proposed_action directly instead of the long combined interaction text.
- Improved development MAE from 1.1678 (original MiniLM input) to 0.6503.
- Grouped-CV OOF MAE: 0.6232.
- Limitation: the encoder does not receive the full interaction history in this representation.
Chunked context + action MiniLM
- Splits formatted_interaction into non-overlapping chunks of up to approximately 200 tokenizer tokens; the final chunk may be shorter.
- Generated 1,684 chunks from 568 training interactions.
- Encodes each chunk into a 384-dimensional vector, then mean-pools chunk embeddings into one context vector per example.
- Concatenates the 384-dimensional context vector with a separate 384-dimensional proposed-action vector, yielding 768 features per example (568 × 768 for training).
- Trained a Logistic Regression classifier using the grouped folds.
- Grouped-CV OOF MAE: 0.5546, an improvement over action-only MiniLM's 0.6232, but still behind TF-IDF.
- Caveat: formatted_interaction may already contain the proposed action; concatenation therefore gives the action extra emphasis rather than creating completely independent sources of information. Equal-weight mean pooling also loses some order and can dilute important details.
Original hybrid model
- Combines TF-IDF and MiniLM-derived features with LightGBM.
- Official development-set MAE: 0.3147; exact accuracy 72.03%.
- Grouped-CV validation and controlled ablations remain pending.
5. Error analysis and limitations
- The hybrid showed strong performance on risk level 1, but weaker recognition of high-risk (4) and critical-risk (5) examples in development-set error analysis.
- The development set contained only two level-5 examples, so conclusions about that category remain highly uncertain.
- Training data is small and imbalanced; performance on the hidden evaluation set may differ.
- Repeated model development on the same development set risks overfitting model-selection decisions to that set.
- Some notebook cells were rerun across Colab sessions; future experiments should record code versions, data order, model parameters, split indices, random seeds, and per-example predictions to ensure reproducibility.
6. Next steps
1. Archive and reproduce Subtask 1 baselines, configurations, plots, fold IDs, and out-of-fold predictions.
2. Move to Subtask 2a (harm-domain multilabel classification) using a simple TF-IDF + One-vs-Rest baseline and grouped validation.
3. Train Subtask 2b (risk-driver multilabel classification) using the same evaluation design.
4. Return to Subtask 1 and test whether predicted harm domains and risk drivers improve severity estimation. For cross-validation experiments, auxiliary labels/features must be generated without leakage from held-out folds; do not feed gold validation/test labels to the severity model.
5. Compare TF-IDF-only, embedding-only, and combined features with the same classifier and the same folds; test ordinal-aware methods if time permits.
7. Repository layout
AgentRisk/
├── subtask1/
│   ├── README.md
│   ├── notebooks/      # Original Colab notebook and experiment notebooks
│   ├── results/        # grouped_cv_results.csv, dev_results.csv
│   ├── predictions/    # Per-example OOF/dev predictions, keyed by example_id
│   ├── figures/        # Confusion matrices and diagnostic charts
│   └── configs/        # Fold assignments, seeds, and hyperparameters
└── subtask2/
Reproducibility reminder: Record dataset version, Python/package versions, seeds, data/preprocessing code, fold assignments, and metrics. Avoid committing access tokens, raw private data, or large model checkpoints to Git.
8. Summary
The initial research phase established a reproducible evaluation strategy and useful baseline results, identified a serious MiniLM input-length limitation, and showed that explicitly representing the proposed action and chunked interaction context improves MiniLM's performance. TF-IDF + Logistic Regression remains the strongest model tested under grouped cross-validation. The development-set hybrid result is promising but requires validation under the same protocol. The project will next investigate whether modeling harm domains and risk drivers helps predict severity on unseen scenarios.