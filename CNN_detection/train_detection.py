from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import warnings
import preprocess_detection
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, MaxPooling2D, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

"""

This file contains implementation of functions for:
    - set up of the cnn model
    - training of the cnn model
    
"""

def build_cnn_model():

    """
     Method for setting up the model

    """
    cnn_model = Sequential()

    cnn_model.add(Conv2D(32, (5, 5), activation="relu", input_shape=(200, 200, 1)))
    cnn_model.add(MaxPool2D((2, 2)))
    cnn_model.add(Dropout(0.2))

    cnn_model.add(Conv2D(64, (3, 3), activation="relu"))
    cnn_model.add(MaxPool2D((2, 2)))
    cnn_model.add(Dropout(0.2))

    cnn_model.add(Conv2D(128, (3, 3), activation="relu"))
    cnn_model.add(MaxPool2D((2, 2)))
    cnn_model.add(Dropout(0.2))

    cnn_model.add(Conv2D(256, (3, 3), activation="relu"))
    cnn_model.add(MaxPool2D((2, 2)))
    cnn_model.add(Dropout(0.2))

    cnn_model.add(Flatten())
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(512, activation="relu"))
    cnn_model.add(Dense(2, activation="softmax"))

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

    cnn_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # print(Model.summary())

    return cnn_model


def train_cnn_model(model):

    """
     Method for training of the model

     Parameters
     ----------
     model : tensorflow model
         Cnn model

     Returns
     -------
     DCNN - detection model

   """

    train_set = preprocess_detection.train
    valid_set = preprocess_detection.valid

    history = model.fit(train_set, validation_data=valid_set,
                        epochs=55, steps_per_epoch=120, verbose=1)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    model.save('medical_detection_cnn_model_v1.h5')

# Launch training

model = build_cnn_model()
train_cnn_model(model)
