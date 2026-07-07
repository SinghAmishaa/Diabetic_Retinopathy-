from pathlib import Path
import cv2

from preprocessing import preprocess_image
from config import TRAIN_IMAGES, PROCESSED_IMAGES

print("TRAIN_IMAGES:", TRAIN_IMAGES)
print("PROCESSED_IMAGES:", PROCESSED_IMAGES)

def process_all_images():
    # Create processed_images folder if it doesn't exist
    PROCESSED_IMAGES.mkdir(exist_ok=True)

    image_paths = list(TRAIN_IMAGES.glob("*.png"))

    print(f"Found {len(image_paths)} images.")

    for i, image_path in enumerate(image_paths, start=1):

        processed = preprocess_image(image_path)

        output_path = PROCESSED_IMAGES / image_path.name

        # Convert back to uint8 before saving
        processed = (processed * 255).astype("uint8")

        processed = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(output_path), processed)

        if i % 100 == 0 or i == len(image_paths):
            print(f"Processed {i}/{len(image_paths)} images")

    print("✅ Dataset preprocessing completed!")


if __name__ == "__main__":
    process_all_images()