# Swahili News Classification

This repository classifies Swahili news articles into six topics with five models, from TF-IDF baselines to a fine-tuned AfriBERTa. The Colab notebook runs the full pipeline and produces every table and figure in the report.

The six topics are *uchumi* (economy), *kitaifa* (national), *michezo* (sports), *kimataifa* (international), *burudani* (entertainment) and *afya* (health). The data is the Swahili News Classification dataset (Davis, 2020), the corpus used in the Zindi AI4D Swahili News Classification Challenge.

## Results

| Model | Selected run | Test macro-F1 | 95% CI | Accuracy | Log loss |
|---|---|---|---|---|---|
| Naive Bayes | `B07` | 0.690 | 0.674–0.704 | 0.868 | 0.623 |
| Logistic Regression | `L05` | 0.817 | 0.805–0.829 | 0.889 | 0.333 |
| BiLSTM | `bilstm_maxpool` | 0.817 | 0.805–0.829 | 0.886 | 0.353 |
| TextCNN | `textcnn_wide` | 0.792 | 0.780–0.805 | 0.871 | 0.423 |
| AfriBERTa | `T04` | **0.868** | 0.857–0.879 | 0.925 | 0.289 |

Test set: 7,303 articles. Per-class scores, significance tests and the error analysis are in the [report](report/report.pdf).

Each model is represented by its best run on validation macro-F1. Macro-F1 is the main metric because the classes are imbalanced 13 to 1. Every run, what it changed and why, is in [`results/experiment_log.csv`](results/experiment_log.csv).

## Run on Google Colab

Open [`notebooks/run_on_colab.ipynb`](notebooks/run_on_colab.ipynb) in Colab, set the runtime to a T4 GPU and run all cells. The notebook clones this repository, installs the requirements, downloads the data and runs each step. The outputs of our own runs are committed, so any training cell can be skipped.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/eda.py
python src/baselines.py
python src/train_neural.py --model bilstm
python src/train_neural.py --model textcnn --dropout 0.5
python src/train_transformer.py
python src/compare_models.py
python src/error_analysis.py
```

`train_neural.py --help` lists the options each BiLSTM and TextCNN run changes. The data (about 40 MB) and fastText vectors (about 230 MB) download on first use into `data/`.

## Repository layout

```
src/
  config.py            paths, labels, seed, device
  data.py              download, remove leaks and duplicates, stratified validation split
  preprocess.py        glued-sentence and ligature repair, cleaning modes
  eda.py               exploratory analysis
  experiment_log.py    one row per run in results/experiment_log.csv
  metrics.py           metrics, confusion matrix and learning-curve plots
  baselines.py         TF-IDF + Naive Bayes, TF-IDF + Logistic Regression
  sequence_data.py     vocabulary, batching, fastText loading
  neural_models.py     BiLSTM with attention or max pooling, TextCNN
  train_neural.py      one BiLSTM or TextCNN run per call
  train_transformer.py AfriBERTa-small fine-tuning
  compare_models.py    test-set comparison and McNemar tests
  error_analysis.py    errors by length, rare words, code-switching; label audit; shortcuts
notebooks/             Colab notebook
results/               experiment log, metrics, test predictions, analysis tables
figures/               every figure in the report
report/                the research report (PDF)
```

## Team

| Member | Work |
|---|---|
| Gentil Tonny Christian Iradukunda | Data pipeline, EDA, AfriBERTa, report |
| Dan Paul Dushime | Naive Bayes, Logistic Regression, evaluation metrics, model comparison |
| Rene Pierre Ntabana | BiLSTM, experiment log, Colab notebook |
| Thierry Alain Tresor Ibyishaka | TextCNN, error analysis |

## References

Davis, D. (2020). *Swahili: News classification dataset* (Version 0.2) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5514203

Ogueji, K., Zhu, Y., & Lin, J. (2021). Small data? No problem! Exploring the viability of pretrained multilingual language models for low-resourced languages. *Proceedings of the 1st Workshop on Multilingual Representation Learning*, 116–126.
