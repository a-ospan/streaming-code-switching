# SwitchLingua: Code-Switching Prediction Project

Team course project, MS in Economics and Data Science, Northeastern University (Fall 2026).
Checkpoints: proposal (done), mid-semester progress presentation, final presentation and report.

## Dataset

- Hugging Face: `Shelton1013/SwitchLingua_text`
- The project MUST cover all 15 language pairs in the dataset (14 X-English plus German-French;
  the paper describes 12 languages, the released text data has 15 pairs). Never drop a pair silently; if a pair is too
  small or problematic, flag it and ask before changing anything.
- Before writing any processing code, inspect the actual schema (column names, how language tags
  are stored, tokenization). Do not assume field names.

## Task definition

Prefix-only multitask prediction. At token position t, using ONLY tokens 1..t (no lookahead):

- `y_sw` (binary): does token t+1 belong to a different language than token t?
- `y_dur` (3 classes, defined only when y_sw = 1): length of the new language segment
  starting at t+1, bucketed as `1-2`, `3-6`, `7+` tokens.

Rules:
- No information from position t+1 or later may leak into features. Treat any leakage as a bug.
- The last token of a sentence has no t+1; exclude it from training targets.
- y_dur is only evaluated on positions where the true y_sw = 1.
- Labeling code must have unit tests on a few hand-checked example sentences.

## Environment

- OS: Windows
- Python environment: conda env named `sad`
- If the shell does not already show `(sad)`, run commands as
  `conda run -n sad python ...` and `conda run -n sad pip install ...`
- Ask before installing new packages, and add any new dependency to `requirements.txt`.
- Use Windows-compatible paths (prefer `pathlib`).

## Repository layout

```
data/          raw and processed data (gitignored)
checkpoints/   saved models (gitignored)
src/           reusable Python modules (loading, labeling, features, models, evaluation)
scripts/       runnable entry points (prepare_data.py, train.py, evaluate.py)
notebooks/     exploration only; reusable logic belongs in src/
tests/         pytest tests
results/       small result tables and figures to commit and share with the team
reports/       presentation and report material
```

## Conventions

- Set and log random seeds for every experiment.
- Split train/val/test stratified by language pair; keep splits fixed once created and save them.
- Save each experiment's config and metrics to `results/` with a clear run name.
- Report metrics per language pair AND overall. For y_sw, class imbalance is expected: report
  precision, recall, F1, and PR-AUC, not just accuracy. For y_dur, report macro-F1 and a
  confusion matrix.
- Always compare against simple baselines (majority class, logistic regression on prefix features)
  before more complex models.

## How to work with me

- Plan first: for any non-trivial change, propose the approach and wait for approval before coding.
- Keep changes small and focused so diffs are easy to review; suggest a commit after each working step.
- Explain modeling choices briefly, since we must defend them in presentations.
- Never delete data or overwrite results without asking.
- Do not use em dashes in any text, comments, or documentation.
