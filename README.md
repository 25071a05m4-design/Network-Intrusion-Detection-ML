# Network Intrusion Detection — ML Challenge

A multiclass machine-learning project for the Network Intrusion Detection ML Challenge. The task is to classify network traffic into four labels: **Benign**, **Availability**, **Reconnaissance**, and **Spoofing**.

The competition is evaluated using **macro F1-score**, so performance on the minority classes matters as much as performance on the majority class.

## Repository structure

```text
Network-Intrusion-Detection-ML/
├── data/                  # Place the Kaggle CSV files here (not committed)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── outputs/               # Generated model and prediction CSVs
├── src/
│   ├── train.py           # Train, validate, and create submission.csv
│   └── eda.py             # Basic dataset inspection
├── .gitignore
└── requirements.txt
```

## 1. Set up in VS Code (Windows)

Open this repository folder in VS Code. In the terminal, run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt and run:

```cmd
.venv\Scripts\activate.bat
```

## 2. Add the competition data

Download the official competition files from Kaggle and place them in the `data/` folder:

- `data/train.csv`
- `data/test.csv`
- `data/sample_submission.csv`

The dataset files are intentionally excluded from Git. Use only the data permitted by the competition rules; do not add external labelled datasets.

## 3. Inspect the data

```bash
python src/eda.py
```

## 4. Train and generate predictions

```bash
python src/train.py
```

The script performs a stratified validation split, reports the validation macro F1-score and per-class metrics, trains a CatBoost multiclass classifier with class balancing, and writes:

- `outputs/submission.csv` — Kaggle submission with columns `id,Target`
- `outputs/catboost_model.cbm` — saved trained model

The script uses the sample submission's ID order when creating the output, where possible. Check the generated CSV and confirm its columns and row count before uploading it to Kaggle.

## Methodology

1. Read the official training and test CSV files.
2. Separate the target label and identifier from the input features.
3. Detect categorical columns and prepare their values for CatBoost.
4. Use a stratified hold-out split to estimate generalization while preserving class proportions.
5. Train a CatBoost multiclass model with balanced class weights to address class imbalance.
6. Evaluate using macro F1 and a per-class classification report.
7. Refit using the full labelled training data and predict the test set.

A random stratified split is a useful baseline, but it may not fully represent the competition's hidden traffic-profile distribution. Treat the validation score as an estimate, not a guarantee of leaderboard performance. Compare alternative validation strategies and models only using the permitted competition data.

## Important

- No competition score is claimed in this repository until you run the code and record the actual result.
- The generated submission contains class labels, not probability columns.
- Review the competition's current submission format and rules before submitting.
- Keep the final code, README, and the Kaggle prediction CSV ready for the required submission form.
