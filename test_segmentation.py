import keras
import tensorflow as tf
import warnings
import utils
import os
from keras.callbacks import CSVLogger
from tensorflow.keras.optimizers import *
import train_segmentation as tr

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ignore unnecessary warnings

warnings.simplefilter("ignore")

SIZE = 128

"""

This file contains implementation of functions for:
    - evaluation of the segmentation model
    - calculate performance metrics

"""

def test_unet_model():
    """
       Method for testing of the unet model

    """

    model = keras.models.load_model('../CNN_segmentation/medical_segmentation_cnn_model.h5',
                                                                custom_objects={'accuracy': tf.keras.metrics.MeanIoU(num_classes=4),
                                                                                "dice_coef": utils.dice_coef,
                                                                                "precision": utils.precision,
                                                                                "sensitivity": utils.sensitivity,
                                                                                "dice_coef_necrotic": utils.dice_coef_nct,
                                                                                "dice_coef_edema": utils.dice_coef_ed,
                                                                                "dice_coef_enhancing": utils.dice_coef_et
                                                                                }, compile=False)

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                          patience=2, min_lr=0.000001, verbose=1),

        ]

    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=4), utils.dice_coef,
                           utils.precision, utils.sensitivity, utils.dice_coef_nct,
                           utils.dice_coef_ed, utils.dice_coef_et])

    print("Evaluate based on test data:")
    results = model.evaluate(tr.test_generator, batch_size=100, callbacks=callbacks)
    print("Test loss, Test accuracy:", results)

# Launch testing

# test_unet_model()