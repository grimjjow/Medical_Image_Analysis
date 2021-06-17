import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.callbacks import CSVLogger
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import warnings
import preprocess_segmentation as ps
import models
import utils
import os
from CNN_segmentation.data_generator import DataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ignore unnecessary warnings

warnings.simplefilter("ignore")

SIZE = 128

"""

This file contains implementation of functions for:
    - data retrieval
    - data generation
    - training of the unet model

"""

input_layer = Input((SIZE, SIZE, 2))

train_and_val_directories = [f.path for f in os.scandir(ps.training_path) if f.is_dir()]

def path_to_image(dirList):

    """
         Method for linking each path to image

         Parameters
         ----------
         dirList : list
             Directories of training and validation images

         Returns
         -------
         list
             list of image directories

    """

    list = []
    for i in range(0, len(dirList)):
        list.append(dirList[i][dirList[i].rfind('/') + 1:])
    return list

train_and_test_ids = path_to_image(train_and_val_directories);

# Launch data split and data generation

train_test_ids, val_ids = train_test_split(train_and_test_ids, test_size=0.2)
train_ids, test_ids = train_test_split(train_test_ids, test_size=0.15)

training_generator = DataGenerator(train_ids)
valid_generator = DataGenerator(val_ids)
test_generator = DataGenerator(test_ids)

def train_unet_model():

    """
         Method for training of the unet model

    """

    model = models.build_2dunet_model(input_layer, 'he_normal', 0.2)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=4), utils.dice_coef, utils.precision,
                           utils.sensitivity,
                           utils.dice_coef_nct, utils.dice_coef_ed, utils.dice_coef_et])


    callbacks = [
          keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=2, min_lr=0.000001, verbose=1),

        ]

    history = model.fit(training_generator,
                        epochs = 35,
                        steps_per_epoch=len(train_ids),
                        callbacks= callbacks,
                        validation_data = valid_generator
                        )

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    model.save('medical_segmentation_cnn_model.h5')

# Launch training

# train_unet_model()

def visualize_training():
    """
         Method visualization of accuracy, loss and dice coef of the model

    """

    # model = keras.models.load_model('../CNN_segmentation/medical_segmentation_cnn_model.h5',
    #                                 custom_objects={'accuracy': tf.keras.metrics.MeanIoU(num_classes=4),
    #                                                 "dice_coef": utils.dice_coef,
    #                                                 "precision": utils.precision,
    #                                                 "sensitivity": utils.sensitivity,
    #                                                 "specificity": utils.specificity,
    #                                                 "dice_coef_necrotic": utils.dice_coef_necrotic,
    #                                                 "dice_coef_edema": utils.dice_coef_edema,
    #                                                 "dice_coef_enhancing": utils.dice_coef_enhancing
    #                                                 }, compile=False)

    loaded_model = pd.read_csv('training_log.log', sep=',', engine='python')

    acc = loaded_model['accuracy']
    val_acc = loaded_model['val_accuracy']

    epoch = range(len(acc))

    loss = loaded_model['loss']
    val_loss = loaded_model['val_loss']

    train_dice = loaded_model['dice_coef']
    val_dice = loaded_model['val_dice_coef']

    f, ax = plt.subplots(1, 3, figsize=(16, 8))

    ax[0].plot(epoch, acc, label='Training Accuracy')
    ax[0].plot(epoch, val_acc, label='Validation Accuracy')
    ax[0].legend()

    ax[1].plot(epoch, loss, label='Training Loss')
    ax[1].plot(epoch, val_loss, label='Validation Loss')
    ax[1].legend()

    ax[2].plot(epoch, train_dice, label='Training dice coef')
    ax[2].plot(epoch, val_dice, label='Validation dice coef')
    ax[2].legend()

    plt.show()

# Launch visualization

visualize_training()
