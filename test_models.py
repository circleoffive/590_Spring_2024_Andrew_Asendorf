#####################################
# test_models.py                    #
#####################################
import numpy as np
import tensorflow as tf
import cv2
import preprocessing
import matplotlib.pyplot as plt
import csv


def dice_coefficient(mask1, mask2):
    # Convert masks to grayscale
    mask1_gray = np.mean(mask1, axis=-1, keepdims=True)
    mask2_gray = np.mean(mask2, axis=-1, keepdims=True)

    # Resize grayscale masks to match the target size
    target_size = (256, 256)
    mask1_resized = cv2.resize(mask1_gray, target_size)
    mask2_resized = cv2.resize(mask2_gray, target_size)

    intersection = np.sum(mask1_resized * mask2_resized)
    union = np.sum(mask1_resized) + np.sum(mask2_resized)
    dice = (2.0 * intersection) / (union + 1e-7)  # Add a small value
    return dice


def iou_coefficient(mask_true, mask_pred):
    intersection = np.logical_and(mask_true, mask_pred)
    union = np.logical_or(mask_true, mask_pred)
    iou = np.sum(intersection) / np.sum(union)
    return iou


def put_csv(model, image_path, dice_score, iou_score):
    csv_file = 'ensemble.csv'
    data = [[dice_score,iou_score, image_path, model]]
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


def predict_model(model, image):
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    mask = model.predict(image)[0]  # Get model prediction
    return mask


def threshold_mask(mask, threshold=0.5):
    binary_mask = (mask > threshold).astype(np.uint8)
    return binary_mask


def preprocess_image(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize pixel values
    return img


def calculate_dice_scores(model, image_paths, mask_paths,
                          model_name='U-Net'):
    dice_scores = []
    iou_scores = []
    for i in range(len(image_paths)):
        # Preprocess the input image
        input_image = preprocess_image(image_paths[i])

        # Preprocess the ground truth mask
        ground_truth_mask = preprocess_image(mask_paths[i])

        # Get the mask output from the model
        mask_output = predict_model(model, input_image)

        # Threshold the model's output mask
        binary_mask_output = threshold_mask(mask_output)

        dice_score = dice_coefficient(ground_truth_mask,
                                      binary_mask_output)
        dice_scores.append(dice_score)

        # Calculate Intersection over Union (IOU)
        iou_score = iou_coefficient(ground_truth_mask,
                                    binary_mask_output)
        iou_scores.append(iou_score)

        # put_csv(model_name, image_paths[i], dice_score, iou_score)
        print("Dice Coefficient:", dice_score, "IOU:", iou_score,
              "Path:", image_paths[i])

    return dice_scores, iou_scores


def ensemble_combined(model_file_names, image_paths, mask_paths,
                      model_name):
    dice_scores = []
    models = []
    iou_scores = []
    # load all models
    for i in range(len(model_file_names)):
        models.append(tf.keras.models.load_model(model_file_names[i]))

    for i in range(len(image_paths)):
        model_predictions = []

        # Preprocess the input image
        input_image = preprocess_image(image_paths[i])

        # Preprocess the ground truth mask
        ground_truth_mask = preprocess_image(mask_paths[i])

        # Get the output from the models
        total_models = len(models)
        for j in range(total_models):
            model_predictions.append(predict_model(models[j],
                                                   input_image))

        # Combine predictions from all models
        combined_predictions = (np.sum(model_predictions, axis=0) /
                                len(models))  # Average predictions
        binary_mask_output = threshold_mask(combined_predictions)

        dice_score = dice_coefficient(ground_truth_mask,
                                      binary_mask_output)
        dice_scores.append(dice_score)

        # Calculate Intersection over Union (IOU)
        iou_score = iou_coefficient(ground_truth_mask,
                                    binary_mask_output)
        iou_scores.append(iou_score)

        # put_csv(model_name, image_paths[i], dice_score, iou_score)
        print("Dice Coefficient:", dice_score, "IOU:", iou_score,
              "Path:", image_paths[i])

    return dice_scores, iou_scores


def calculate_mean_dice(dice_scores):
    mean_dice = np.mean(dice_scores)
    return mean_dice


def calculate_median_dice(dice_scores):
    median_dice = np.median(dice_scores)
    return median_dice


def model_comparison(choice):
    # put_csv('model', 'image_path', 'dice_score', 'iou_score')

    dice_scores = []
    iou_scores = []
    model_file_names = []

    # Load the saved model
    if choice == 1 or choice == 2:
        # No augmentation
        model_file_names.append("Brain2_10_20_25_No_aug.keras")
        # with augmentation
        model_file_names.append("model_Standard_U-Net_noDrop.gh5")
        # with dropout
        model_file_names.append("model_Standard_dropout_allAug.gh5")

    if choice == 2:
        # attention
        model_file_names.append("model_Attention_allAug.gh5")
        # Res U-Net
        model_file_names.append("model_Res_allAug.gh5")
        # Attention Res
        model_file_names.append("model_Attention_Res_U-Net_allAug.gh5")

    if choice == 2 or choice == 3:
        # U-Net++
        model_file_names.append("model_plus_plus_allAug.gh5")

    testing_folder = "split_data\\test"
    image_paths, mask_paths = preprocessing.get_paths(testing_folder)

    if choice == 1:
        labels = ['Without', 'Augmentation', 'Both']
    if choice == 2:
        labels = ['no Aug', 'no Drop', 'Standard', 'Attention',
                  'Residual', 'Att + Res', 'U-Net++']
    if choice == 3:
        labels = ['U-Net++', 'All', 'top-3']

    for i in range(len(model_file_names)):
        model = tf.keras.models.load_model(model_file_names[i])
        dice_one, iou_one = (calculate_dice_scores
                             (model, image_paths, mask_paths,
                              model_name=labels[i]))
        dice_scores.append(dice_one)
        iou_scores.append(iou_one)

    if choice == 3:
        all_models = ["Brain2_10_20_25_No_aug.keras",
                      "model_Standard_U-Net_noDrop.gh5",
                      "model_Standard_dropout_allAug.gh5",
                      "model_Attention_allAug.gh5",
                      "model_Res_allAug.gh5",
                      "model_Attention_Res_U-Net_allAug.gh5",
                      "model_plus_plus_allAug.gh5"]
        dice_one, iou_one = ensemble_combined(all_models,
                                              image_paths,
                                              mask_paths,
                                              model_name='all')
        print('dice: ', dice_one)
        print('iou: ',iou_one)
        dice_scores.append(dice_one)
        iou_scores.append(iou_one)

        top_three = ["model_Res_allAug.gh5",
                     "model_plus_plus_allAug.gh5",
                     "model_Attention_Res_U-Net_allAug.gh5"]
        dice_one, iou_one = ensemble_combined(top_three, image_paths,
                                              mask_paths,
                                              model_name='top-3')
        print('dice: ', dice_one)
        print('iou: ',iou_one)
        dice_scores.append(dice_one)
        iou_scores.append(iou_one)

    for i in range(len(dice_scores)):
        print(labels[i])
        print('Mean Dice: ', calculate_mean_dice(dice_scores[i]))
        print('Median Dice: ', calculate_median_dice(dice_scores[i]))
        print('\n')
        print('Mean IoU: ', calculate_mean_dice(iou_scores[i]))
        print('Median IoU: ', calculate_median_dice(iou_scores[i]))
        print('\n')

    # Create box plot
    plt.figure(figsize=(8, 6))
    # plt.boxplot(data, labels=labels)
    plt.boxplot(iou_scores, labels=labels)

    plt.title('IoU Comparison')
    # plt.xlabel('Algorithms')
    plt.ylabel('IoU')
    plt.grid(True)
    plt.show()

    # Create box plot
    plt.figure(figsize=(8, 6))
    # plt.boxplot(data, labels=labels)
    plt.boxplot(dice_scores, labels=labels)

    plt.title('Dice Comparison')
    # plt.xlabel('Algorithms')
    plt.ylabel('Dice Coefficient')
    plt.grid(True)
    plt.show()
