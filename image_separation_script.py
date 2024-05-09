#####################################
# image_separation_script.py        #
#####################################

# NOTE: This script is ONLY used to clean the dataset and separated
# into training, validation and testing folders
# this code was NOT used in creating or testing AI models

import os
import re
from PIL import Image
import shutil
import random


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in
            re.split('(\d+)', s)]


def is_mask_not_blank(mask_path):
    mask = Image.open(mask_path)
    width, height = mask.size

    for x in range(width):
        for y in range(height):
            pixel = mask.getpixel((x, y))
            # Check if any RGB value is not 0
            if pixel != 0:
                return True
    return False


def load_images_and_masks(root_folder):
    image_paths = []
    mask_paths = []

    # Iterate through each folder in the root folder
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            # Get all TIFF files in the folder
            tiff_files = [file for file in os.listdir(folder_path)
                          if file.endswith('.tif') or
                          file.endswith('.tiff')]
            # Sort the TIFF files using natural sorting
            tiff_files.sort(key=natural_sort_key)
            # Iterate through sorted TIFF files
            for i in range(0, len(tiff_files), 2):
                image_file = tiff_files[i]
                mask_file = tiff_files[i+1]
                image_path = os.path.join(folder_path, image_file)
                mask_path = os.path.join(folder_path, mask_file)
                if is_mask_not_blank(mask_path):
                    image_paths.append(image_path)
                    mask_paths.append(mask_path)
                    print(image_path)
                    print(mask_path)

    return image_paths, mask_paths


def split_data(image_paths, mask_paths, output_folder):
    # Create output folders if they don't exist
    training_images_folder = os.path.join(output_folder,
                                          'training_images')
    training_masks_folder = os.path.join(output_folder,
                                         'training_masks')
    validation_images_folder = os.path.join(output_folder,
                                            'validation_images')
    validation_masks_folder = os.path.join(output_folder,
                                           'validation_masks')
    test_images_folder = os.path.join(output_folder,
                                      'test_images')
    test_masks_folder = os.path.join(output_folder,
                                     'test_masks')

    for folder in [training_images_folder, training_masks_folder,
                   validation_images_folder, validation_masks_folder,
                   test_images_folder, test_masks_folder]:
        os.makedirs(folder, exist_ok=True)

    # Calculate number of images for each split
    total_images = len(image_paths)
    num_training = int(total_images * 0.8)
    num_validation = int(total_images * 0.1)
    num_test = total_images - num_training - num_validation

    # Shuffle the image and mask paths
    combined = list(zip(image_paths, mask_paths))
    random.shuffle(combined)
    image_paths_shuffled, mask_paths_shuffled = zip(*combined)

    # Copy images and masks to their respective folders
    # based on the split
    for i in range(total_images):
        image_path = image_paths_shuffled[i]
        mask_path = mask_paths_shuffled[i]

        if i < num_training:
            shutil.copy(image_path, training_images_folder)
            shutil.copy(mask_path, training_masks_folder)
        elif i < num_training + num_validation:
            shutil.copy(image_path, validation_images_folder)
            shutil.copy(mask_path, validation_masks_folder)
        else:
            shutil.copy(image_path, test_images_folder)
            shutil.copy(mask_path, test_masks_folder)

    print("Data splitting and copying completed.")

image_paths, mask_paths = load_images_and_masks('dataset')
output_folder = 'Test_split\\split_data'
split_data(image_paths, mask_paths, output_folder)