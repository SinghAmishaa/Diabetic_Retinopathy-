import pandas as pd
from config import TRAIN_CSV


def load_dataset():
    """
    Load the APTOS training dataset.
    """
    df = pd.read_csv(TRAIN_CSV)
    return df


def dataset_summary(df):
    """Print basic information about the dataset."""

    print("=" * 50)
    print("Dataset Summary")
    print("=" * 50)

    print(f"Shape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nClass Distribution:")
    print(df["diagnosis"].value_counts().sort_index())


if __name__ == "__main__":

    df = load_dataset()

    dataset_summary(df)