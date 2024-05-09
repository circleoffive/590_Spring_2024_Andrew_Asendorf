#####################################
# Standard_u_net.py                 #
#####################################
from keras.metrics import MeanIoU
import keras
import tensorflow as tf
import preprocessing


def conv_block(x, num_filter, size, dropout, batch_norm=False):

    if batch_norm:
        conv = tf.keras.layers.Conv2D(num_filter, (size, size),
                                      kernel_initializer='he_normal',
                                      padding="same")(x)
        conv = tf.keras.layers.BatchNormalization(axis=3)(conv)
        conv = tf.keras.layers.Activation("relu")(conv)
    else:
        conv = tf.keras.layers.Conv2D(num_filter, (size, size),
                                      activation='relu',
                                      kernel_initializer='he_normal',
                                      padding="same")(x)

    if dropout > 0:
        conv = tf.keras.layers.Dropout(dropout)(conv)

    return conv


def encoder_block(x, num_filter, size, dropout, batch_norm=False):
    en_conv = conv_block(x, num_filter, size, dropout, batch_norm)
    en_conv = conv_block(en_conv, num_filter, size, 0,
                         batch_norm)

    return en_conv


def decoder_block(x, concat, num_filters, size, dropout, batch_norm):

    u = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2),
                                        strides=(2, 2),
                                        padding='same')(x)
    u = tf.keras.layers.concatenate([u, concat])
    de_conv = conv_block(u, num_filters, size, dropout, batch_norm)
    de_conv = conv_block(de_conv, num_filters, size, 0,
                         batch_norm)

    return de_conv


def create_u_net(aug=False, drop=False):

    # size of the images
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    IMG_CHANNELS = 3

    # 32 filters then multiples by 2 every level down
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

    # Augmentation or no Augmentation
    if aug:
        train_gen = preprocessing.create_gens(train_df,
                                              aug_dict=tr_aug_dict)
    else:
        train_gen = preprocessing.create_gens(train_df, aug_dict={})

    valid_gen = preprocessing.create_gens(valid_df, aug_dict={})

    if drop:
        dropout_one = 0.1
        dropout_two = 0.2
        dropout_three = 0.3
    else:
        dropout_one = 0
        dropout_two = 0
        dropout_three = 0

    # Input Layer
    inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT,
                                    IMG_CHANNELS))

    lambda_inputs = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
    batch_on_off = False

    # MaxPooling down samples the features and the Channels will be
    # doubled after every MaxPooling
    # 1st Contraction path Layer (Encoder)
    c1 = encoder_block(x=lambda_inputs,num_filter=filter_one,size=3,
                       dropout=dropout_one,batch_norm=batch_on_off)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    # 2nd Contraction path Layer (Encoder)
    c2 = encoder_block(x=p1,num_filter=filter_two,size=3,
                       dropout=dropout_one,batch_norm=batch_on_off)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    # 3rd Contraction path Layer (Encoder)
    c3 = encoder_block(x=p2,num_filter=filter_three,size=3,
                       dropout=dropout_two,batch_norm=batch_on_off)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    # 4th Contraction path Layer (Encoder)
    c4 = encoder_block(x=p3,num_filter=filter_four,size=3,
                       dropout=dropout_two,batch_norm=batch_on_off)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

    # Bottom Layer - "Bottleneck"
    c5 = encoder_block(x=p4, num_filter=filter_five, size=3,
                       dropout=dropout_three, batch_norm=batch_on_off)

    # 1st Expansive path Layer (Decoder)
    c6 = decoder_block(x=c5, concat=c4, size=3,
                       num_filters=filter_four, dropout=dropout_two,
                       batch_norm=batch_on_off)

    # 2nd Expansive path Layer (Decoder)
    c7 = decoder_block(x=c6, concat=c3, size=3,
                       num_filters=filter_three,
                       dropout=dropout_two, batch_norm=batch_on_off)

    # 3rd Expansive path Layer (Decoder)
    c8 = decoder_block(x=c7, concat=c2, size=3,
                       num_filters=filter_two,
                       dropout=dropout_one, batch_norm=batch_on_off)

    # 4th Expansive path Layer (Decoder)
    c9 = decoder_block(x=c8, concat=c1, size=3,
                       num_filters=filter_one,
                       dropout=dropout_one, batch_norm=batch_on_off)

    # output layer
    outputs = tf.keras.layers.Conv2D(1, (1, 1),
                                     activation='sigmoid')(c9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=[keras.metrics.BinaryIoU
                           (target_class_ids=(0, 1), threshold=0.5,
                            name=None, dtype=None)
                           ])

    model.summary()

    # set the batch size and epoches
    batch_size = 20
    epochs = 20

    model_name = 'new_model_standard_U_Net'
    model_file_name = model_name + '.gh5'

    # Define the callbacks
    checkpointer = (tf.keras.callbacks.ModelCheckpoint
                    (model_file_name, verbose=1, save_best_only=True))
    early_stopping = (tf.keras.callbacks.EarlyStopping
                      (patience=3, monitor='val_loss'))

    callbacks = [checkpointer, early_stopping,
                 tf.keras.callbacks.TensorBoard(log_dir='logs')]

    # the total number of augmented images generated per epoch
    rotation_number = (1 + tr_aug_dict['rotation_range']
                       // rotation_range )
    width_shift_number = (1 + tr_aug_dict['width_shift_range']
                          // width_shift_range)
    height_shift_number = (1 + tr_aug_dict['height_shift_range']
                           // height_shift_range)
    shear_range_number = (1 + tr_aug_dict['shear_range']
                          // shear_range)
    zoom_number = (1 + tr_aug_dict['zoom_range'] // zoom_range)
    total_augmented_images = (len(train_df) *
                              rotation_number *
                              width_shift_number * \
                             height_shift_number *
                              shear_range_number * \
                             zoom_number * 2)

    # Calculate the new number of steps per epoch based on the batch
    # size and total augmented images
    if not aug:
        total_augmented_images = len(train_df)
    new_steps_per_epoch = total_augmented_images // batch_size

    history = model.fit(train_gen,
                        steps_per_epoch=new_steps_per_epoch,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=valid_gen,
                        validation_steps=len(valid_df) / batch_size)

    print("Model is complete!")