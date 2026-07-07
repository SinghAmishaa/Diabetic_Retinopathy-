import pandas as pd
from sklearn.model_selection import train_test_split

from config import TRAIN_CSV


def split_dataset():

    df = pd.read_csv(TRAIN_CSV)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["diagnosis"]
    )

    train_df.to_csv("dataset/train_split.csv", index=False)
    val_df.to_csv("dataset/val_split.csv", index=False)

    print("Training Images:", len(train_df))
    print("Validation Images:", len(val_df))


if __name__ == "__main__":
    split_dataset()