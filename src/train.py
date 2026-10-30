from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_PATH = DATA_DIR / "sample_submission.csv"
TARGET = "Target"
RANDOM_STATE = 42


def find_id_column(frame: pd.DataFrame) -> str | None:
    for column in frame.columns:
        if column.lower() == "id":
            return column
    return None


def prepare_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_id = find_id_column(train)
    test_id = find_id_column(test)

    excluded = {TARGET}
    if train_id is not None:
        excluded.add(train_id)

    feature_columns = [c for c in train.columns if c not in excluded]

    missing_in_test = sorted(set(feature_columns) - set(test.columns))
    if missing_in_test:
        raise ValueError(f"Test data is missing feature columns: {missing_in_test}")

    x_train = train[feature_columns].copy()
    x_test = test[feature_columns].copy()

    # CatBoost handles numeric missing values natively. Categorical values
    # are converted to strings and missing entries receive an explicit token.
    categorical_columns = x_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    for column in categorical_columns:
        x_train[column] = x_train[column].astype("string").fillna("__MISSING__").astype(str)
        x_test[column] = x_test[column].astype("string").fillna("__MISSING__").astype(str)

    # Ensure numeric columns have compatible numeric dtypes.
    for column in x_train.columns:
        if column not in categorical_columns:
            x_train[column] = pd.to_numeric(x_train[column], errors="coerce")
            x_test[column] = pd.to_numeric(x_test[column], errors="coerce")

    return x_train, x_test, categorical_columns


def main() -> None:
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            "Expected data/train.csv and data/test.csv. Download the official "
            "competition files and place them in the data/ folder."
        )

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    if TARGET not in train.columns:
        raise ValueError(f"Training data must contain the target column '{TARGET}'.")

    if train[TARGET].isna().any():
        raise ValueError("Target column contains missing labels; inspect the data first.")

    x_train, x_test, categorical_columns = prepare_features(train, test)
    y = train[TARGET].astype(str)

    print("Training rows:", len(train))
    print("Test rows:", len(test))
    print("Number of features:", x_train.shape[1])
    print("Categorical features:", categorical_columns)
    print("\nClass distribution:")
    print(y.value_counts())

    x_fit, x_valid, y_fit, y_valid = train_test_split(
        x_train,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        auto_class_weights="Balanced",
        iterations=1800,
        learning_rate=0.05,
        depth=8,
        random_seed=RANDOM_STATE,
        verbose=200,
        allow_writing_files=False,
    )

    model.fit(
        x_fit,
        y_fit,
        cat_features=categorical_columns,
        eval_set=(x_valid, y_valid),
        early_stopping_rounds=150,
        verbose=200,
    )

    valid_predictions = np.asarray(model.predict(x_valid)).reshape(-1).astype(str)
    macro_f1 = f1_score(y_valid, valid_predictions, average="macro")
    print(f"\nValidation Macro F1: {macro_f1:.6f}")
    print("\nPer-class validation report:")
    print(classification_report(y_valid, valid_predictions, zero_division=0))

    # Refit a fresh model on all labelled rows, using the best iteration
    # found during validation (with a minimum of one iteration).
    best_iterations = model.get_best_iteration()
    final_iterations = best_iterations + 1 if best_iterations >= 0 else 1800

    final_model = CatBoostClassifier(
        loss_function="MultiClass",
        auto_class_weights="Balanced",
        iterations=final_iterations,
        learning_rate=0.05,
        depth=8,
        random_seed=RANDOM_STATE,
        verbose=200,
        allow_writing_files=False,
    )
    final_model.fit(x_train, y, cat_features=categorical_columns)

    test_predictions = np.asarray(final_model.predict(x_test)).reshape(-1).astype(str)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(OUTPUT_DIR / "catboost_model.cbm"))

    sample = pd.read_csv(SAMPLE_PATH) if SAMPLE_PATH.exists() else None
    test_id_column = find_id_column(test)

    if sample is not None and "id" in sample.columns:
        if test_id_column is not None:
            prediction_frame = pd.DataFrame(
                {"id": test[test_id_column].values, TARGET: test_predictions}
            )
            submission = sample[["id"]].merge(
                prediction_frame,
                on="id",
                how="left",
                validate="one_to_one",
            )
            if submission[TARGET].isna().any():
                raise ValueError(
                    "Some sample_submission IDs did not match test IDs. "
                    "Check the identifier columns and row alignment."
                )
        else:
            if len(sample) != len(test_predictions):
                raise ValueError("Sample submission and test row counts do not match.")
            submission = sample[["id"]].copy()
            submission[TARGET] = test_predictions
    else:
        if test_id_column is not None:
            submission = pd.DataFrame(
                {"id": test[test_id_column].values, TARGET: test_predictions}
            )
        else:
            submission = pd.DataFrame(
                {"id": np.arange(len(test_predictions)), TARGET: test_predictions}
            )

    submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)
    print("\nSaved:", OUTPUT_DIR / "submission.csv")
    print("Submission shape:", submission.shape)
    print("Submission columns:", submission.columns.tolist())
    print("\nPredicted class distribution:")
    print(submission[TARGET].value_counts())


if __name__ == "__main__":
    main()
