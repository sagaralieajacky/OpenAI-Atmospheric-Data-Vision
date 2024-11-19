import cv2
import os
import numpy as np

def preprocess_image(image_path):
    """Preprocesses an image for AI model input."""
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (256, 256))
    normalized_image = resized_image / 255.0
    return normalized_image

def preprocess_dataset(folder_path):
    """Preprocesses all images in a folder."""
    processed_images = []
    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)
        if os.path.isfile(full_path):
            processed_images.append(preprocess_image(full_path))
    return np.array(processed_images)

# Example usage
if __name__ == "__main__":
    images = preprocess_dataset('sample_data/')
    print(f"Processed {len(images)} images.")
