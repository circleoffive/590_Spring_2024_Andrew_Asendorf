#####################################
# test_models.py                    #
#####################################

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import test_models


def image_overlay(img_path, mask_path, model_mask):

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Load mask as grayscale
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Ensure mask is in the correct format (CV_8UC1)
    mask = cv2.threshold(mask, 0, 255,
                         cv2.THRESH_BINARY)[1]  # Binarize mask
    # Convert mask to RGB for overlaying
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # Load mask as grayscale
    model_mask = cv2.threshold(model_mask, 0, 255,
                               cv2.THRESH_BINARY)[1]
    # Convert mask to RGB for overlaying
    model_mask_rgb = cv2.cvtColor(model_mask, cv2.COLOR_GRAY2RGB)

    # Find contours using cv.findContours() with
    # cv.RETR_EXTERNAL and cv.CHAIN_APPROX_SIMPLE
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contours_model, _ = cv2.findContours(model_mask,
                                         cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask with the same dimensions as the input image
    contour_mask = np.zeros_like(mask)

    # Draw contours on the blank mask with white color (255)
    # and thickness 1
    cv2.drawContours(contour_mask, contours, -1,255, 1)

    # Overlay the contour mask on the
    # input image with transparency effect
    together = cv2.addWeighted(img, 1,
                               cv2.cvtColor(contour_mask,
                                            cv2.COLOR_GRAY2RGB),
                               0.8,
                               0)

    # Fill contours in the model_mask with a specified color
    # (e.g., red)
    filled_model_mask = np.zeros_like(model_mask_rgb)
    # Fill contours with red color
    cv2.fillPoly(filled_model_mask, contours_model, (255, 0, 0))

    # Overlay the filled model_mask on the input
    # image with transparency effect
    together_filled = cv2.addWeighted(together, 1,
                                      filled_model_mask, 0.7, 0)

    return img, mask_rgb, model_mask_rgb, together_filled


def ground_truth_overlay(img_path, mask_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load mask as grayscale
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Ensure mask is in the correct format (CV_8UC1)
    mask = cv2.threshold(mask, 0, 255,
                         cv2.THRESH_BINARY)[1]  # Binarize mask
    # Convert mask to RGB for overlaying
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # Find contours using cv.findContours() with cv.RETR_EXTERNAL
    # and cv.CHAIN_APPROX_SIMPLE
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask with the same dimensions as the input image
    contour_mask = np.zeros_like(mask)

    # Draw contours on the blank mask with white color (255)
    # and thickness 1
    cv2.drawContours(contour_mask, contours, -1, 255, 1)

    # Overlay the contour mask on the input image with
    # transparency effect
    together = cv2.addWeighted(img, 1,
                               cv2.cvtColor(contour_mask,
                                            cv2.COLOR_GRAY2RGB), 0.8, 0)

    return mask_rgb, together


# display images does a 1x4 display
def display_images(img, mask_rgb, model_mask_rgb, together_filled):
    num_images = len(img)
    num_rows = num_images + 1
    num_cols = 4
    plt.figure(figsize=(7, 10))

    for i in range(num_images):

        plt.subplot(num_rows, num_cols, i*num_cols+1)
        if i == 0:
            plt.title('Image')
        plt.imshow(img[i])
        plt.axis('off')

        plt.subplot(num_rows, num_cols, i*num_cols+2)
        if i == 0:
            plt.title('Ground Truth')
        plt.imshow(mask_rgb[i])
        plt.axis('off')

        plt.subplot(num_rows, num_cols, i*num_cols+3)
        if i == 0:
            plt.title('Model')
        plt.imshow(model_mask_rgb[i])
        plt.axis('off')

        # Plot the final image with contours filled in the model_mask
        plt.subplot(num_rows, num_cols, i*num_cols+4)
        if i == 0:
            plt.title('Overlay')
        plt.imshow(together_filled[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def display_comparison_images(ground_truth_masks, ground_overlays,
                              model_masks, overlay_images):
    num_images = len(ground_truth_masks)
    num_models = len(model_masks)
    row = 0
    col = 0
    plt.rcParams['figure.figsize'] = [12,16]
    # using the variable axs for multiple Axes
    fig, axs = plt.subplots(8, 6)

    # Plot ground truth masks and overlays in the first row
    print("work")
    for i in range(num_images):

        axs[row, i*2].imshow(ground_truth_masks[i])
        axs[row, i*2].axis('off')
        axs[row, i*2+1].imshow(ground_overlays[i])
        axs[row, i*2+1].axis('off')
    row = 1
    for i in range(num_models):
        if i != 0 and (i % 7) == 0:
            row = 1
            col = col + 1
        axs[row, col*2].imshow(model_masks[i])
        axs[row, col*2].axis('off')
        # axs[0, i*2].title('Ground Truth')
        axs[row, col*2+1].imshow(overlay_images[i])
        axs[row, col*2+1].axis('off')
        # axs[0, i*2+1].title('Overlay')
        row = row + 1

    plt.tight_layout(h_pad=.08, w_pad=.1)
    plt.show()


# Function to preprocess input image
def preprocess_image(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize pixel values
    return img


# Function to obtain mask output from the model
def get_mask(model, image):
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    mask = model.predict(image)[0]  # Get model prediction
    return mask


# Threshold function to create binary mask with sharp edges
def threshold_mask(mask, threshold=0.5):
    binary_mask = (mask > threshold).astype(np.uint8)
    return binary_mask


def ensemble_image(model_file_names, image):
    model_predictions = []
    models = []
    # load all models
    for i in range(len(model_file_names)):
        models.append(tf.keras.models.load_model(model_file_names[i]))

    # Get the output from the models
    total_models = len(models)
    for j in range(total_models):
        model_predictions.append(get_mask(models[j], image))

    # Combine predictions from all models
    combined_predictions = (np.sum(model_predictions, axis=0) /
                            len(models))  # Average predictions
    binary_mask_output = threshold_mask(combined_predictions)

    return binary_mask_output


def show_model_output(choice=2):
    brain_images = []
    ground_truth_masks = []
    model_masks = []
    overlay_images = []

    paths_ground_truths = []
    paths_input_images = []

    top_3_models = ["model_Res_allAug.gh5",
                    "model_plus_plus_allAug.gh5",
              "model_Attention_Res_U-Net_allAug.gh5"]

    # Load the saved model
    model_file_name = "model_plus_plus_allAug.gh5"
    model = tf.keras.models.load_model(model_file_name)

    # Input image path
    paths_input_images.append("split_data\\test\\test_images"
                              "\\TCGA_DU_6400_19830518_24.tif")
    paths_input_images.append("split_data\\test\\test_images"
                              "\\TCGA_HT_7882_19970125_20.tif")
    paths_input_images.append("split_data\\test\\test_images"
                              "\\TCGA_HT_7877_19980917_26.tif")
    paths_input_images.append("split_data\\test\\test_images"
                              "\\TCGA_DU_7010_19860307_40.tif")
    paths_input_images.append("split_data\\test\\test_images"
                              "\\TCGA_DU_A5TW_19980228_20.tif")

    # Ground truth mask path
    paths_ground_truths.append("split_data\\test\\test_masks\\"
                               "TCGA_DU_6400_19830518_24_mask.tif")
    paths_ground_truths.append("split_data\\test\\test_masks\\"
                               "TCGA_HT_7882_19970125_20_mask.tif")
    paths_ground_truths.append("split_data\\test\\test_masks\\"
                               "TCGA_HT_7877_19980917_26_mask.tif")
    paths_ground_truths.append("split_data\\test\\test_masks\\"
                               "TCGA_DU_7010_19860307_40_mask.tif")
    paths_ground_truths.append("split_data\\test\\test_masks\\"
                               "TCGA_DU_A5TW_19980228_20_mask.tif")

    for i in range(len(paths_input_images)):
        # Preprocess the input image
        input_image = preprocess_image(paths_input_images[i])

        # Preprocess the ground truth mask
        ground_truth_mask = preprocess_image(paths_ground_truths[i])

        # Get the mask output from the model
        if choice == 1:
            mask_output = get_mask(model, input_image)
        else:
            mask_output = ensemble_image(top_3_models, input_image)

        # Threshold the model's output mask
        binary_mask_output = threshold_mask(mask_output)

        dice_score = test_models.dice_coefficient(ground_truth_mask,
                                                  binary_mask_output)
        print(dice_score)

        (img,
         mask_rgb,
         model_mask_rgb,
         together_filled) = image_overlay(paths_input_images[i],
                                          paths_ground_truths[i],
                                          binary_mask_output)
        brain_images.append(img)
        ground_truth_masks.append(mask_rgb)
        model_masks.append(model_mask_rgb)
        overlay_images.append(together_filled)

    display_images(brain_images,
                   ground_truth_masks,
                   model_masks,
                   overlay_images)


def image_comparisons():
    models = []
    paths_ground_truths = []
    paths_input_images = []
    input_images = []
    ground_truth_masks = []
    ground_overlays = []
    model_masks = []
    overlay_images = []


    # models


    # Input image path
    paths_input_images.append("split_data\\test\\test_images\\"
                              "TCGA_DU_7294_19890104_23.tif")
    paths_input_images.append("split_data\\test\\test_images\\"
                              "TCGA_DU_6405_19851005_44.tif")
    paths_input_images.append("split_data\\test\\test_images\\"
                              "TCGA_HT_7877_19980917_26.tif")

    # Ground truth mask path
    paths_ground_truths.append("split_data\\test\\test_masks\\"
                               "TCGA_DU_7294_19890104_23_mask.tif")
    paths_ground_truths.append("split_data\\test\\test_masks\\"
                               "TCGA_DU_6405_19851005_44_mask.tif")
    paths_ground_truths.append("split_data\\test\\test_masks\\"
                               "TCGA_HT_7877_19980917_26_mask.tif")

    for i in range(len(paths_input_images)):
        # Preprocess the input image
        input_images.append(preprocess_image(paths_input_images[i]))

        # Preprocess the ground truth mask
        ground_truth_masks.append(preprocess_image
                                  (paths_ground_truths[i]))

    for i in range(len(model_file_names)):
        models.append(tf.keras.models.load_model(model_file_names[i]))
    for i in range(len(input_images)):
        # create overlay for ground truth
        mask, overlay = ground_truth_overlay(paths_input_images[i],
                                             paths_ground_truths[i])
        ground_overlays.append(overlay)
        for j in range(len(model_file_names)):
            # Get the mask output from the model
            mask_output = get_mask(models[j], input_images[i])

            # Threshold the model's output mask
            binary_mask_output = threshold_mask(mask_output)

            dice_score = (test_models.dice_coefficient
                          (ground_truth_masks[i], binary_mask_output))
            print(dice_score)

            (img,
             mask_rgb,
             model_mask_rgb,
             together_filled) = image_overlay(paths_input_images[i],
                                              paths_ground_truths[i],
                                              binary_mask_output)

            model_masks.append(model_mask_rgb)
            overlay_images.append(together_filled)

    display_comparison_images(ground_truth_masks,
                              ground_overlays,
                              model_masks,
                              overlay_images)