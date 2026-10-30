from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "train.csv"
TEST_PATH = ROOT / "data" / "test.csv"


def main() -> None:
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            "Place train.csv and test.csv inside the project's data/ folder."
        )

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    print("\nTRAIN SHAPE:", train.shape)
    print("\nTEST SHAPE:", test.shape)
    print("\nTRAIN COLUMNS:")
    print(train.columns.tolist())

    if "Target" in train.columns:
        print("\nTARGET COUNTS:")
        print(train["Target"].value_counts(dropna=False))
        print("\nTARGET PROPORTIONS:")
        print(train["Target"].value_counts(normalize=True, dropna=False))

    print("\nMISSING VALUES IN TRAIN (top 20):")
    print(train.isna().sum().sort_values(ascending=False).head(20))

    print("\nDATA TYPES:")
    print(train.dtypes.value_counts())

    train_features = set(train.columns) - {"Target"}
    test_features = set(test.columns)
    print("\nFEATURES ONLY IN TRAIN:", sorted(train_features - test_features))
    print("FEATURES ONLY IN TEST:", sorted(test_features - train_features))


if __name__ == "__main__":
    main()
