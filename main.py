#####################################
# main.py                           #
#####################################
#
# Andrew Asendorf's Graduate Project
# California State University Dominguez Hills
# Spring 2024
# Title: Automatic Biomedical Image Segmentation for Brain Tumors using
# U-Net Architecture
# Advisors:
# Cedar Sinai Project: Biomedical Image Segmentation and Classification
# Turned in: May 1, 2024

import standard_u_net
import attention_u_net
import residual_u_net
import att_res_u_net
import u_net_plus_plus
import test_models
import model_output


def welcome_print():
    print("Hello, This is Andrew Asendorf's Graduate Project!")
    print('Title: Automatic Biomedical Image Segmentation '
          'for Brain Tumors using U-Net Architecture\n')


def options():
    print('This program creates U-Net models, compares previously created '
          'U-Net Models, and demonstrates the output of U-Net models.')
    print('Select one of the following options:')
    print('(1) Create a new model')
    print('(2) Compare previously made models')
    print('(3) Demonstrate outputs of U-Net models')
    print('(4) Exit Program')

    return user_choice(4)


def user_choice(number_of_choices):
    while True:
        try:
            choice = int(input(f'Enter your choice (1-{number_of_choices}): '))
            if choice in range(1, number_of_choices+1):
                print('\n')
                return choice
            else:
                print(f'Error: Please enter a number between 1 and {number_of_choices}.')
        except ValueError:
            print('Error: Please enter a valid integer.')


def make_model():
    print('You have selected to make a new model.')
    print('Select one of the following options:')
    print('(1) Standard U-Net')
    print('(2) Attention U-Net')
    print('(3) Residual U-Net')
    print('(4) Attention Residual U-Net')
    print('(5) U-Net++')
    print('(6) Back to main menu')
    choice = user_choice(6)

    if choice == 1:
        print('You have selected to create the Standard U-Net Model.')
        print('Select one of the following options:')
        print('(1) Standard U-Net without Augmentation and Dropout')
        print('(2) Standard U-Net with Augmentation and without Dropout')
        print('(3) Standard U-Net with Augmentation and Dropout')
        print('(4) Back to main menu')
        choice_two = user_choice(5)
        if choice_two == 1:
            print('You chose to create a Standard U-Net model without Augmentation and Dropout')
            standard_u_net.create_u_net(aug=False, drop=False)
        if choice_two == 2:
            print('You chose to create a Standard U-Net model with '
                  'Augmentation and without Dropout')
            standard_u_net.create_u_net(aug=True, drop=False)
        if choice_two == 3:
            print('You chose to create a Standard U-Net model with '
                  'Augmentation and Dropout')
            standard_u_net.create_u_net(aug=True, drop=True)
        if choice_two == 4:
            return
    if choice == 2:
        print('You chose to create an Attention U-Net model.')
        attention_u_net.create_attention_u_net()
    if choice == 3:
        print('You chose to create a Residual U-Net model.')
        residual_u_net.create_residual_u_net()
    if choice == 4:
        print('You chose to create an Attention Residual U-Net model.')
        att_res_u_net.create_att_res_u_net()
    if choice == 5:
        print('You chose to create a U-Net++ model.')
        u_net_plus_plus.create_u_net_plus_plus()


def compare_models():
    print('You have selected compare previously made models.')
    print('Select one of the following options:')
    print('(1) Compare Standard U-Net with/without Augmentation and Dropout')
    print('(2) Compare 7 different single U-Net models')
    print('(3) Compare 2 Ensemble methods and U-Net++')
    print('(4) Back to main menu')
    choice = user_choice(4)
    if choice == 4:
        return
    test_models.model_comparison(choice)


def model_pictures():
    print('You have selected to see the outputs of U-Net models.')
    print('Select one of the following options:')
    print('(1) Output of 7 different single U-Net models')
    print('(2) Output of Top-3 combined models using the Ensemble Method')
    print('(3) Back to main menu')
    choice = user_choice(3)
    if choice == 3:
        return
    if choice == 1:
        model_output.image_comparisons()
    if choice == 2:
        model_output.show_model_output()


if __name__ == '__main__':
    welcome_print()
    while True:
        choice = options()
        if choice == 1:
            make_model()
        if choice == 2:
            compare_models()
        if choice == 3:
            model_pictures()
        if choice == 4:
            break
        print('Select one of the following options:')
        print('(1) Choose another option')
        print('(2) Exit Program')
        choice = user_choice(2)
        if choice == 2:
            break

    print('Have a nice day!')
