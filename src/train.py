print("train.py started")

from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt

from config import (
    TRAIN_SPLIT,
    VAL_SPLIT,
    PROCESSED_IMAGES,
    MODELS_DIR,
    EPOCHS,
    BATCH_SIZE
)

from dataset_loader import load_dataset
from model import build_cnn
from resnet50_model import build_resnet50

MODEL_NAME = "resnet50"   

def train():

    print("Loading datasets...")

    train_dataset = load_dataset(
        TRAIN_SPLIT,
        PROCESSED_IMAGES,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_dataset = load_dataset(
        VAL_SPLIT,
        PROCESSED_IMAGES,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    print(f"Building {MODEL_NAME}...")

    if MODEL_NAME == "cnn":
       model = build_cnn()
    elif MODEL_NAME == "resnet50":
       model = build_resnet50()
    else:
       raise ValueError("Invalid model name")

    MODELS_DIR.mkdir(exist_ok=True)
    
    Path("outputs").mkdir(exist_ok=True)

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ModelCheckpoint(
            MODELS_DIR / f"{MODEL_NAME}.keras",
            save_best_only=True
        )

    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    plot_history(history)


def plot_history(history):

    plt.figure(figsize=(10,5))

    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

    plt.legend()

    plt.title(f"{MODEL_NAME.upper()} Accuracy")
    plt.tight_layout()
    plt.savefig(f"outputs/{MODEL_NAME}_accuracy.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(10,5))

    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")

    plt.legend()

    plt.title(f"{MODEL_NAME.upper()} Loss")
    plt.tight_layout()
    plt.savefig(f"outputs/{MODEL_NAME}_loss.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    train()