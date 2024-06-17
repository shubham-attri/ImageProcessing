import cv2
import os
import glob
import numpy as np
import time
from numba import cuda

# Ensure the results directory exists
os.makedirs('results', exist_ok=True)

def log_message(message):
    """
    Log messages to a log file in the results directory.
    """
    with open('results/log.txt', 'a') as log_file:
        log_file.write(f"{message}\n")

@cuda.jit
def sobel_filter(input_image, output_image):
    x, y = cuda.grid(2)
    if x >= 1 and x < input_image.shape[0] - 1 and y >= 1 and y < input_image.shape[1] - 1:
        Gx = (input_image[x-1, y-1] - input_image[x+1, y-1]) + (2 * (input_image[x-1, y] - input_image[x+1, y])) + (input_image[x-1, y+1] - input_image[x+1, y+1])
        Gy = (input_image[x-1, y-1] + (2 * input_image[x, y-1]) + input_image[x+1, y-1]) - (input_image[x-1, y+1] + (2 * input_image[x, y+1]) + input_image[x+1, y+1])
        output_image[x, y] = np.sqrt(Gx**2 + Gy**2)

@cuda.jit
def gaussian_blur(input_image, output_image):
    x, y = cuda.grid(2)
    if x >= 1 and x < input_image.shape[0] - 1 and y >= 1 and y < input_image.shape[1] - 1:
        output_image[x, y] = (
            4 * input_image[x, y] +
            2 * (input_image[x-1, y] + input_image[x+1, y] + input_image[x, y-1] + input_image[x, y+1]) +
            input_image[x-1, y-1] + input_image[x-1, y+1] + input_image[x+1, y-1] + input_image[x+1, y+1]
        ) / 16

@cuda.jit
def canny_edge_detection(input_image, output_image):
    x, y = cuda.grid(2)
    if x >= 1 and x < input_image.shape[0] - 1 and y >= 1 and y < input_image.shape[1] - 1:
        Gx = (input_image[x-1, y-1] - input_image[x+1, y-1]) + (2 * (input_image[x-1, y] - input_image[x+1, y])) + (input_image[x-1, y+1] - input_image[x+1, y+1])
        Gy = (input_image[x-1, y-1] + (2 * input_image[x, y-1]) + input_image[x+1, y-1]) - (input_image[x-1, y+1] + (2 * input_image[x, y+1]) + input_image[x+1, y+1])
        magnitude = np.sqrt(Gx**2 + Gy**2)
        output_image[x, y] = 255 if magnitude > 100 else 0

def process_image(file_path):
    """
    Process a single image: apply three transformations using CUDA and save the results.
    """
    try:
        # Read the image
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to read {file_path}")

        # Allocate device memory
        d_image = cuda.to_device(image)
        d_sobel = cuda.device_array_like(image)
        d_blur = cuda.device_array_like(image)
        d_canny = cuda.device_array_like(image)

        # Define block and grid sizes
        threads_per_block = (16, 16)
        blocks_per_grid_x = int(np.ceil(image.shape[0] / threads_per_block[0]))
        blocks_per_grid_y = int(np.ceil(image.shape[1] / threads_per_block[1]))
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        # Perform transformations
        transformations = [
            ('transformation1', sobel_filter, d_sobel),
            ('transformation2', gaussian_blur, d_blur),
            ('transformation3', canny_edge_detection, d_canny)
        ]

        for trans_name, trans_func, d_output in transformations:
            start_time = time.time()
            trans_func[blocks_per_grid, threads_per_block](d_image, d_output)
            transformed_image = d_output.copy_to_host()
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
