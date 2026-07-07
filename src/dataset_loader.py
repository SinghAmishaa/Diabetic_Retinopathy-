import pandas as pd
import tensorflow as tf

from config import IMAGE_SIZE

AUTOTUNE = tf.data.AUTOTUNE


def load_dataset(csv_path, image_dir, batch_size=32, shuffle=True):
    """
    Creates a TensorFlow Dataset from a CSV file.
    """

    df = pd.read_csv(csv_path)

    image_paths = [
        str(image_dir / f"{img}.png")
        for img in df["id_code"]
    ]

    labels = df["diagnosis"].values

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    def load_image(path, label):

        image = tf.io.read_file(path)

        image = tf.image.decode_png(
            image,
            channels=3
        )

        image = tf.image.resize(
            image,
            IMAGE_SIZE
        )

        image = image / 255.0

        return image, label

    dataset = dataset.map(
        load_image,
        num_parallel_calls=AUTOTUNE
    )

    if shuffle:
        dataset = dataset.shuffle(1000)

    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset