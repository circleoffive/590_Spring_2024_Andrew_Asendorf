#####################################
# preprocessing.py                  #
#####################################
import os
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
import re


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in
            re.split('(\d+)', s)]


def get_paths(root_folder):
    image_paths = []
    mask_paths = []

    # Iterate through each folder in the root folder
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            # Get all TIFF files in the folder
            tiff_files = [file for file in os.listdir(folder_path) if
                          file.endswith('.tif') or
                          file.endswith('.tiff')]
            # Sort the TIFF files using natural sorting
            tiff_files.sort(key=natural_sort_key)
            # Iterate through sorted TIFF files
            for i in range(0, len(tiff_files)):
                file = tiff_files[i]
                path = os.path.join(folder_path, file)
                # Differentiate between image and mask files
                if 'mask' in file.lower():
                    mask_paths.append(path)
                else:
                    image_paths.append(path)

    return image_paths, mask_paths


def create_df(root_folder):
    image_paths, mask_paths = get_paths(root_folder)

    df = pd.DataFrame(data={'images_paths': image_paths,
                            'masks_paths': mask_paths})

    return df


def create_gens(df, aug_dict):
    img_size = (256, 256)
    batch_size = 20

    img_gen = ImageDataGenerator(**aug_dict)
    msk_gen = ImageDataGenerator(**aug_dict)

    # Create general generator
    image_gen = img_gen.flow_from_dataframe(df, x_col='images_paths',
                                            class_mode=None,
                                            color_mode='rgb',
                                            target_size=img_size,
                                            batch_size=batch_size,
                                            save_to_dir=None,
                                            save_prefix='image',
                                            seed=1)

    mask_gen = msk_gen.flow_from_dataframe(df, x_col='masks_paths',
                                           class_mode=None,
                                           color_mode='grayscale',
                                           target_size=img_size,
                                           batch_size=batch_size,
                                           save_to_dir=None,
                                           save_prefix= 'mask',
                                           seed=1)

    gen = zip(image_gen, mask_gen)

    for (img, msk) in gen:
        img = img / 255
        msk = msk / 255
        msk[msk > 0.5] = 1
        msk[msk <= 0.5] = 0

        yield img, msk
