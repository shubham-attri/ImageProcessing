import cv2
import os
import glob
import numpy as np
import time

# Ensure the results directory exists
os.makedirs('results', exist_ok=True)

def log_message(message):
    """
    Log messages to a log file in the results directory.
    """
    with open('results/log.txt', 'a') as log_file:
        log_file.write(f"{message}\n")

def process_image(file_path):
    """
    Process a single image: apply three transformations and save the results.
    """
    try:
        # Read the image
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to read {file_path}")

        # Determine the number of channels in the image
        num_channels = 1 if len(image.shape) == 2 else image.shape[2]

        # Define transformations
        transformations = []

        # Add Sobel filter as the first transformation
        transformations.append(('transformation1', lambda img: cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=5)))
        
        # Add other transformations regardless of the number of channels
        transformations.extend([
            ('transformation2', lambda img: cv2.GaussianBlur(img, (5, 5), 0)),
            ('transformation3', lambda img: cv2.Canny(img, 100, 200))
        ])

        # Apply each transformation and save the result
        for trans_name, trans_func in transformations:
            start_time = time.time()
            transformed_image = trans_func(image)
            transformed_path = file_path.replace('data/images', f'data/{trans_name}')
            os.makedirs(os.path.dirname(transformed_path), exist_ok=True)
            cv2.imwrite(transformed_path, transformed_image)
            end_time = time.time()
            elapsed_time = end_time - start_time
            log_message(f"{file_path} -> {trans_name} processed in {elapsed_time:.5f} seconds")

    except Exception as e:
        log_message(f"Error processing {file_path}: {e}")

def main():
    """
    Main function to process all .tiff images in the data/images directory.
    """
    image_files = glob.glob('data/images/*.tiff')
    if not image_files:
        log_message("No images found in data/images/")
        return

    for image_file in image_files:
        process_image(image_file)

if __name__ == "__main__":
    main()

