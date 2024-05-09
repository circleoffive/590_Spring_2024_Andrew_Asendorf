#####################################
# u_net_plus_plus.py                #
#####################################
from keras.metrics import MeanIoU
import keras
import tensorflow as tf
import preprocessing


def conv_block(x, num_filter, size, dropout):

    conv = tf.keras.layers.Conv2D(num_filter, (size,size),
                                  activation='relu',
                                  kernel_initializer='he_normal',
                                  padding="same")(x)
    if dropout > 0:
        conv =tf.keras.layers.Dropout(dropout)(conv)
    conv = tf.keras.layers.Conv2D(num_filter, (size,size),
                                  activation='relu',
                                  kernel_initializer='he_normal',
                                  padding="same")(conv)

    return conv


def create_u_net_plus_plus():
    # image size
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    IMG_CHANNELS = 3

    filter_one = 32
    filter_two = filter_one * 2
    filter_three = filter_two * 2
    filter_four = filter_three * 2
    filter_five = filter_four * 2

    train_df = preprocessing.create_df("split_data\\training")
    valid_df = preprocessing.create_df("split_data\\validation")

    rotation_range = 0.1  # 10%
    width_shift_range = 0.05  # 5%
    height_shift_range = 0.05  # 5%
    shear_range = 0.05  # 5%
    zoom_range = 0.05  # 5%
    horizontal_flip = True
    vertical_flip = True
    fill_mode = 'nearest'

    tr_aug_dict = dict(rotation_range=rotation_range,
                                width_shift_range=width_shift_range,
                                height_shift_range=height_shift_range,
                                shear_range=shear_range,
                                zoom_range=zoom_range,
                                horizontal_flip=horizontal_flip,
                                vertical_flip=vertical_flip,
                                fill_mode=fill_mode)

    train_gen = preprocessing.create_gens(train_df,
                                          aug_dict=tr_aug_dict)
    valid_gen = preprocessing.create_gens(valid_df, aug_dict={})

    inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT,
                                    IMG_CHANNELS))

    lambda_inputs = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    # MaxPooling down samples the features and the Channels will be
    # doubled after every MaxPooling
    # 1st Contraction path Layer
    c1_1 = conv_block(x=lambda_inputs, num_filter=filter_one, size=3,
                      dropout=0)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1_1)

    # 2nd Contraction path Layer
    c2_1 = conv_block(x=p1, num_filter=filter_two, size=3, dropout=0)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2_1)

    # Nested Dense Convolutional Blocks
    u1_2 = tf.keras.layers.Conv2DTranspose(filter_one, (2, 2),
                                           strides=(2, 2),
                                           padding='same')(c2_1)
    u1_2 = tf.keras.layers.concatenate([u1_2, c1_1], axis=3)
    c1_2 = conv_block(x=u1_2, num_filter=filter_one, size=3,
                      dropout=0)

    # 3rd Contraction path Layer
    c3_1 = conv_block(x=p2, num_filter=filter_three, size=3,
                      dropout=0)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3_1)

    # Nested Dense Convolutional Blocks
    u2_2 = tf.keras.layers.Conv2DTranspose(filter_two, (2, 2),
                                           strides=(2, 2),
                                           padding='same')(c3_1)
    u2_2 = tf.keras.layers.concatenate([u2_2, c2_1], axis=3)
    c2_2 = conv_block(x=u2_2, num_filter=filter_two, size=3,
                      dropout=0)

    # Nested Dense Convolutional Blocks
    u1_3 = tf.keras.layers.Conv2DTranspose(filter_one, (2, 2),
                                           strides=(2, 2),
                                           padding='same')(c2_2)
    u1_3 = tf.keras.layers.concatenate([u1_3, c1_1, c1_2],
                                       axis=3)
    c1_3 = conv_block(x=u1_3, num_filter=filter_one, size=3,
                      dropout=0)

    # 4th Contraction path Layer
    c4_1 = conv_block(x=p3, num_filter=filter_four, size=3, dropout=0)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4_1)

    # Nested Dense Convolutional Blocks
    u3_2 = tf.keras.layers.Conv2DTranspose(filter_three, (2, 2),
                                           strides=(2, 2),
                                           padding='same')(c4_1)
    u3_2 = tf.keras.layers.concatenate([u3_2, c3_1], axis=3)
    c3_2 = conv_block(x=u3_2, num_filter=filter_three, size=3,
                      dropout=0)

    # Nested Dense Convolutional Blocks
    u2_3 = tf.keras.layers.Conv2DTranspose(filter_two, (2, 2),
                                           strides=(2, 2),
                                           padding='same')(c3_2)
    u2_3 = tf.keras.layers.concatenate([u2_3, c2_1, c2_2],
                                       axis=3)
    c2_3 = conv_block(x=u2_3, num_filter=filter_two, size=3,
                      dropout=0)

    # Nested Dense Convolutional Blocks
    u1_4 = tf.keras.layers.Conv2DTranspose(filter_one, (2, 2),
                                           strides=(2, 2),
                                           padding='same')(c2_3)
    u1_4 = tf.keras.layers.concatenate([u1_4, c1_1, c1_2, c1_3],
                                       axis=3)
    c1_4 = conv_block(x=u1_4, num_filter=filter_one, size=3,
                      dropout=0)

    # bottom
    c5_1 = conv_block(x=p4, num_filter=filter_five, size=3,
                      dropout=0.5)

    # 1st Expansive path Layer
    u4_2 = tf.keras.layers.Conv2DTranspose(filter_four, (2, 2),
                                           strides=(2, 2),
                                           padding='same')(c5_1)
    u4_2 = tf.keras.layers.concatenate([u4_2, c4_1], axis=3)
    c4_2 = conv_block(x=u4_2, num_filter=filter_four, size=3,
                      dropout=0)

    # 2nd Expansive path Layer
    u3_3 = tf.keras.layers.Conv2DTranspose(filter_three, (2, 2),
                                           strides=(2, 2),
                                           padding='same')(c4_2)
    u3_3 = tf.keras.layers.concatenate([u3_3, c3_1, c3_2],
                                       axis=3)
    c3_3 = conv_block(x=u3_3, num_filter=filter_three, size=3,
                      dropout=0)

    # 3rd Expansive path Layer
    u2_4 = tf.keras.layers.Conv2DTranspose(filter_two, (2, 2),
                                           strides=(2, 2),
                                           padding='same')(c3_3)
    u2_4 = tf.keras.layers.concatenate([u2_4, c2_1, c2_2, c2_3],
                                       axis=3)
    c2_4 = conv_block(x=u2_4, num_filter=filter_two, size=3,
                      dropout=0)

    # 4th Expansive path Layer
    u1_5 = tf.keras.layers.Conv2DTranspose(filter_one, (2, 2),
                                           strides=(2, 2),
                                           padding='same')(c2_4)
    u1_5 = tf.keras.layers.concatenate([u1_5, c1_1, c1_2, c1_3, c1_4],
                                       axis=3)
    c1_5 = conv_block(x=u1_5, num_filter=filter_one, size=3,
                      dropout=0)

    outputs = tf.keras.layers.Conv2D(1, (1, 1),
                                     activation='sigmoid')(c1_5)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=[keras.metrics.BinaryIoU
                           (target_class_ids=(0, 1),
                            threshold=0.5,
                            name=None,
                            dtype=None)
                           ])

    model.summary()

    batch_size = 20
    epochs = 20

    model_name = 'new_model_U-Net_plus_plus'
    model_file_name = model_name + '.gh5'

    # Define the callbacks
    checkpointer = (tf.keras.callbacks.ModelCheckpoint
                    (model_file_name, verbose=1, save_best_only=True))
    early_stopping = (tf.keras.callbacks.EarlyStopping
                      (patience=3, monitor='val_loss'))

    callbacks = [checkpointer, early_stopping,
                 tf.keras.callbacks.TensorBoard(log_dir='logs')]

    # the total number of augmented images generated per epoch
    rotation_number = (1 + tr_aug_dict['rotation_range'] //
                       rotation_range )
    width_shift_number = (1 + tr_aug_dict['width_shift_range'] //
                          width_shift_range)
    height_shift_number = (1 + tr_aug_dict['height_shift_range'] //
                           height_shift_range)
    shear_range_number = (1 + tr_aug_dict['shear_range'] //
                          shear_range)
    zoom_number = (1 + tr_aug_dict['zoom_range'] // zoom_range)
    total_augmented_images = (len(train_df) *
                              rotation_number *
                              width_shift_number * \
                             height_shift_number *
                              shear_range_number * \
                             zoom_number * 2)

    # Calculate the new number of steps per epoch based on the
    # batch size and total augmented images
    new_steps_per_epoch = total_augmented_images // batch_size

    history = model.fit(train_gen,
                        steps_per_epoch=new_steps_per_epoch,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=valid_gen,
                        validation_steps=len(valid_df) / batch_size)

    print("The model is complete!")
