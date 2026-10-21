# streaming-code-switching

Predictive multitask learning for streaming code-switching, using the SwitchLingua corpus.
Team 10, CS 6140 Machine Learning, Northeastern University (Fall 2026):
Artykzhan Ospan, Mahamane Bachir Sadou, Mukaila Rafiu.

## Task

At every token *t* of an utterance, using **only tokens 1..t**, predict:

| Target | Type | Definition |
|---|---|---|
| `y_sw` | binary | 1 if token *t+1* is in a different language than token *t* |
| `y_dur` | 3 classes | when `y_sw = 1`: length of the new language segment, binned 1-2, 3-6 or 7+ tokens |

No information from position *t+1* or later may reach the inputs. This is enforced by automated tests.

## Data

[`Shelton1013/SwitchLingua_text`](https://huggingface.co/datasets/Shelton1013/SwitchLingua_text) (gated on Hugging Face;
pinned revision `6e048d27`): 234,172 conversations, 974,994 sentences, **15 language pairs** (14 X-English plus
German-French). Hindi-English is reported as two strata (Devanagari and romanized), giving 16 evaluation strata.

## Results so far (validation split, macro average over 16 strata)

| Model | Switch F1 | PR-AUC | Precision | Recall | sigma of F1 | Duration macro-F1 |
|---|---|---|---|---|---|---|
| Majority class | 0.000 | 0.114 | 0.000 | 0.000 | n/a | 0.202 |
| Logistic regression (prefix features) | **0.555** | **0.528** | 0.550 | 0.571 | 0.076 | **0.531** |

Per-stratum results: `results/runs/logreg_val_seed13/per_strata.csv`. Report: `reports/update1_acm/Team10_ProjectUpdate1_Report.pdf`.

## Repository layout

```
src/                   reusable modules
  data.py              load the corpus (pinned revision) and flatten rows into sentences
  tokens.py            model-independent tokenizer (one token per Chinese character)
  lid.py               token-level language ID: script, word-frequency and context rules
  labels.py            y_sw and y_dur
  splits.py            80/10/10 splits by conversation, stratified, duplicates grouped
  streaming.py         prefix-only alignment of words to a model's subwords (no lookahead)
  dataset.py           PyTorch dataset, batching, temperature sampling over pairs
  features.py          prefix-only baseline features
  metrics.py           per-stratum and macro metrics, sigma of F1, confusion matrix
  model.py             causal dual-head predictor (XLM-R with causal mask, or XGLM)
  resources/           romanized Hindi word list (built from the corpus)
scripts/               runnable entry points
  prepare_data.py      tokenize, tag, label and split the whole corpus
  baselines.py         majority class and logistic regression baselines
  train.py             transformer training (GPU; see notebooks/train_colab.ipynb)
  check_alignment.py   tokenizer alignment report
  lid_check.py         hand-validation sheets for the language tags, and their scoring
  build_roman_hindi_lexicon.py, make_report_figures.py, make_paper_figures.py
tests/                 57 pytest tests (labels, splits, no-lookahead, features, model, metrics)
notebooks/
  weeks4-6_pipeline.ipynb   the whole weeks 4-6 pipeline in one self-contained notebook
  train_colab.ipynb         transformer training on a Colab GPU
results/               small result files: splits, data summary, baseline runs, alignment
reports/               Update 1 report (ACM LaTeX source and PDF), slides, working reports
```

## How to run

Requires Python 3.10 or later and a Hugging Face account that has accepted the dataset's terms.

```bash
pip install -r requirements.txt
huggingface-cli login                 # once; the dataset is gated
python -m scripts.prepare_data        # about 10 minutes; reuses the committed splits
python -m scripts.baselines           # about 2 minutes; writes results/runs/
python -m pytest -q tests             # 57 tests
```

Or open `notebooks/weeks4-6_pipeline.ipynb` (Colab or Jupyter) and run all cells: it reproduces the splits, the
data statistics and both baselines.

To rebuild the report: `python -m scripts.make_paper_figures`, then compile `reports/update1_acm/main.tex`
(pdflatex, bibtex, pdflatex twice) or upload that folder to Overleaf.

## Conventions

- Splits are fixed (`results/splits/row_splits.csv.gz`, seed 13) and never regenerated.
- Metrics are reported per stratum and as macro averages, so Cantonese-English (64% of sentences) cannot dominate.
- The test split is kept untouched until the final evaluation.
