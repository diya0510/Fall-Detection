import cv2
import numpy as np
import os

def resize_image(image, size=(224, 224)):
    return cv2.resize(image, size)
def normalize_image(image):
    return image.astype(np.float32) / 255.0

def process_images(input_dir, output_dir, size=(224, 224)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Resize and normalize the image
            img_resized = resize_image(img, size)
            img_normalized = normalize_image(img_resized)

            # Save the processed image
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, (img_normalized * 255).astype(np.uint8))

input_dir = r'C:\Users\Diya\OneDrive\Documents\College\Projects\Fall\fall_dataset\images\train'
output_dir = r'C:\Users\Diya\OneDrive\Documents\College\Projects\Fall\fall_dataset\images\train_resized'
process_images(input_dir, output_dir)
