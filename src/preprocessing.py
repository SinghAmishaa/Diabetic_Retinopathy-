"""
Image preprocessing functions for Diabetic Retinopathy Detection.
"""

import cv2
import numpy as np
from pathlib import Path

from config import IMAGE_SIZE


def load_image(image_path):
    """
    Load an image using OpenCV and convert BGR to RGB.
    """
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def resize_image(image):
    """
    Resize image to the required input size.
    """
    return cv2.resize(image, IMAGE_SIZE)


def normalize_image(image):
    """
    Normalize pixel values between 0 and 1.
    """
    return image.astype(np.float32) / 255.0


def apply_clahe(image):
    """
    Apply CLAHE to improve image contrast.
    """

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))

    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return image


def preprocess_image(image_path):
    """
    Complete preprocessing pipeline.
    """

    image = load_image(image_path)

    image = resize_image(image)

    image = apply_clahe(image)

    image = normalize_image(image)

    return image

from config import TRAIN_IMAGES

if __name__ == "__main__":

    # Get the first image from the dataset
    image_path = next(TRAIN_IMAGES.glob("*.png"))

    print(f"Testing image: {image_path.name}")

    processed_image = preprocess_image(image_path)

    print("Image processed successfully!")
    print("Image Shape:", processed_image.shape)
    print("Data Type:", processed_image.dtype)
    print("Minimum Pixel Value:", processed_image.min())
    print("Maximum Pixel Value:", processed_image.max())