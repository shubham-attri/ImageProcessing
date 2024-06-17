
import cv2
import os
import glob
import numpy as np
from numba import cuda
import time

def log_message(message):
    with open('results/log.txt', 'a') as log_file:
        log_file.write(f"{message}\n")

@cuda.jit
def grayscale_kernel(input_image, output_image):
    x, y = cuda.grid(2)
    if x < output_image.shape[0] and y < output_image.shape[1]:
        r = input_image[x, y, 0]
        g = input_image[x, y, 1]
        b = input_image[x, y, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        output_image[x, y] = gray

@cuda.jit
def gaussian_blur_kernel(input_image, output_image):
    x, y = cuda.grid(2)
    kernel_size = 5
    kernel = np.array([1, 4, 7, 4, 1], dtype=np.float32) / 16
    if x >= kernel_size // 2 and y >= kernel_size // 2 and x < input_image.shape[0] - kernel_size // 2 and y < input_image.shape[1] - kernel_size // 2:
        sum = 0
        for i in range(kernel_size):
            for j in range(kernel_size):
                sum += input_image[x + i - kernel_size // 2, y + j - kernel_size // 2] * kernel[i] * kernel[j]
        output_image[x, y] = sum

def process_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        log_message(f"Failed to read {file_path}")
        return

    transformations = [
        ('transformation1', grayscale_kernel),
        ('transformation2', gaussian_blur_kernel)
    ]

    for trans_name, trans_func in transformations:
        start_time = time.time()
        
        # Prepare input and output arrays
        input_image = cuda.to_device(image)
        output_image = cuda.device_array((image.shape[0], image.shape[1]), dtype=np.float32)
        
        # Define grid and block dimensions
        threads_per_block = (16, 16)
        blocks_per_grid_x = int(np.ceil(image.shape[0] / threads_per_block[0]))
        blocks_per_grid_y = int(np.ceil(image.shape[1] / threads_per_block[1]))
        
        # Launch the kernel
        trans_func[(blocks_per_grid_x, blocks_per_grid_y), threads_per_block](input_image, output_image)
        
        # Copy the result back to the host
        result_image = output_image.copy_to_host().astype(np.uint8)
        
        # Save the processed image
        processed_path = file_path.replace('data/images', f'data/{trans_name}')
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        cv2.imwrite(processed_path, result_image)
        
        end_time = time.time()
        log_message(f"{file_path} -> {trans_name} processed in {end_time - start_time:.2f} seconds")

def main():
    image_files = glob.glob('data/images/*.tiff')
    if not image_files:
        log_message("No images found in data/images/")
        return

    for image_file in image_files:
        process_image(image_file)

if __name__ == "__main__":
    main()
