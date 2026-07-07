import pandas as pd
import matplotlib.pyplot as plt

from config import TRAIN_CSV


def dataset_info():
    df = pd.read_csv(TRAIN_CSV)

    print("=" * 50)
    print("Dataset Information")
    print("=" * 50)

    print("\nShape:")
    print(df.shape)

    print("\nMissing Values:")
    print(df.isnull().sum())

    counts = df["diagnosis"].value_counts().sort_index()

    print("\nClass Distribution:")
    print(counts)

    plt.figure(figsize=(8,5))

    plt.bar(counts.index.astype(str), counts.values)

    plt.title("Diabetic Retinopathy Class Distribution")

    plt.xlabel("Diagnosis")

    plt.ylabel("Number of Images")

    plt.savefig("outputs/class_distribution.png")

    plt.show()


if __name__ == "__main__":
    dataset_info()