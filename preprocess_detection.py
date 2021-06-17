import cv2
import imutils
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import os.path
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np

"""

This file contains implementation of functions for:
    - data retrieval
    - cropping and normalization
    - plotting data and ground truth
    - data generation
    - data split
    
"""

# ignore unnecessary warnings
warnings.filterwarnings("ignore")

# data paths
negative_data_path = Path("../CNN_detection/input/OASIS_detection/no")
positive_data_path = Path("../CNN_detection/input/OASIS_detection/yes")
prediction_data_path = Path("../CNN_detection/input/OASIS_detection/pred")

# image paths
negative_path = list(negative_data_path.glob(r"*.jpg"))
positive_path = list(positive_data_path.glob(r"*.jpg"))
prediction_path = list(prediction_data_path.glob(r"*.jpg"))

def get_training_data(negative_path, positive_path):

    """
      Method for retrieving of initial training data and ground truth

      Parameters
      ----------
      negative_path : os.path
          Path for images of brains with no tumors
      positive_path : os.path
          Path for images of brains with tumors

      Returns
      -------
      training
          dataset of training images and ground truth

    """

    main_data = []

    for neg in negative_path:
        main_data.append(neg)

    for pos in positive_path:
        main_data.append(pos)

    ground_truth = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], main_data))

    images = pd.Series(main_data, name="JPG").astype(str)
    categories = pd.Series(ground_truth, name="Category")

    training = pd.concat([images, categories], axis=1)

    return training


def get_prediction_data(prediction_path):

    """
      Method for retrieving of initial testing data and ground truth

      Parameters
      ----------
      prediction_path : os.path
          Path for images of brains for testing

      Returns
      -------
      testing
          dataset of testing images and ground truth

    """

    ground_truth = list(map(lambda x: os.path.split(os.path.split(x)[0])[1],prediction_path))

    images = pd.Series(prediction_path,name="JPG").astype(str)
    categories = pd.Series(ground_truth,name="Category")

    testing = pd.concat([images,categories],axis=1)

    return testing

def get_main_prediction_data(prediction_path):

    """
      Method for retrieving of initial testing data(images)

      Parameters
      ----------
      prediction_path : os.path
          Path for images of brains for testing

      Returns
      -------
      main_testing
          dataset of testing images without ground truth

    """

    images = pd.Series(prediction_path,name="JPG").astype(str)
    main_testing = pd.DataFrame({"JPG":images})

    return main_testing

def plot_data(negative_path, positive_path):

    """
      Method for plotting intial training data

      Parameters
      ----------
      negative_path : os.path
          Path for images of brains with no tumors
      positive_path : os.path
          Path for images of brains with tumors

    """

    training_data = get_training_data(negative_path, positive_path).sample(frac=1).reset_index(drop=True)

    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10),
                            subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(plt.imread(training_data["JPG"][i]))
        ax.set_title(training_data["Category"][i])
    plt.tight_layout()
    plt.show()

    # fig.show()
    fig.savefig('../CNN_detection/plots_and_outputs/detection_ground_truth.png')

def split_data(negative_path, positive_path):

    """
      Method for splitting the data

      Parameters
      ----------
      negative_path : os.path
          Path for images of brains with no tumors
      positive_path : os.path
          Path for images of brains with tumors

      Returns
      -------
      train_data
          dataset of training images
      test_data
          dataset of testing images

    """

    training_data = get_training_data(negative_path, positive_path).sample(frac=1).reset_index(drop=True)

    train_data,test_data = train_test_split(training_data,train_size=0.9,random_state=42)

    return train_data,test_data

def cropping(image_set, pixels=0):

  processed = []
  for img in image_set:
    gray1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.GaussianBlur(gray1, (5, 5), 0)

    thresh1 = cv2.threshold(gray2, 45, 255, cv2.THRESH_BINARY)[1]
    thresh2 = cv2.erode(thresh1, None, iterations=2)
    thresh3 = cv2.dilate(thresh2, None, iterations=2)

    contour1 = cv2.findContours(thresh3.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour2 = imutils.grab_contours(contour1)
    contour3 = max(contour2, key=cv2.contourArea)

    left = tuple(contour3[contour3[:, :, 0].argmin()][0])
    right = tuple(contour3[contour3[:, :, 0].argmax()][0])
    top = tuple(contour3[contour3[:, :, 1].argmin()][0])
    bottom = tuple(contour3[contour3[:, :, 1].argmax()][0])

    new_img = img[top[1]-pixels:bottom[1]+pixels, left[0]-pixels:right[0]+pixels].copy()
    new_img = cv2.resize(new_img, (224, 224))
    processed.append(new_img)

  return np.array(processed)

def data_generation(train_data, test_data):

    """
      Method for data generation

      Parameters
      ----------
      train_data : dataframe
          Training data
      test_data : dataframe
          Testing data

      Returns
      -------
      Training, validation and testing datasets

    """

    basic_generator = ImageDataGenerator(rescale=1./255,
                                         validation_split=0.1)

    training_set = basic_generator.flow_from_dataframe(dataframe=train_data,
                                                   x_col="JPG",
                                                   y_col="Category",
                                                   color_mode="grayscale",
                                                   class_mode="categorical",
                                                   subset="training",
                                                   batch_size=20,
                                                   target_size=(200,200))

    validation_set = basic_generator.flow_from_dataframe(dataframe=train_data,
                                                         x_col="JPG",
                                                         y_col="Category",
                                                         color_mode="grayscale",
                                                         class_mode="categorical",
                                                         subset="validation",
                                                         batch_size=20,
                                                         target_size=(200,200))

    testing_set = basic_generator.flow_from_dataframe(dataframe=test_data,
                                                  x_col="JPG",
                                                  y_col="Category",
                                                  color_mode="grayscale",
                                                  class_mode="categorical",
                                                  batch_size=20,
                                                  target_size=(200,200))

    main_generator = ImageDataGenerator(rescale=1./255)

    pred = get_main_prediction_data(prediction_path)

    main_testing = main_generator.flow_from_dataframe(dataframe=pred,
                                                           x_col="JPG",
                                                           y_col=None,
                                                           color_mode="grayscale",
                                                            class_mode=None,
                                                            batch_size=20,
                                                            target_size=(200,200))
    return training_set, validation_set, testing_set, pred, main_testing

def data_augmentation(train_data, test_data):

    """
         Method for data generation with data augmentation

     Parameters
     ----------
     train_data : dataframe
         Training data
     test_data : dataframe
         Testing data

     Returns
     -------
     Training, validation and testing datasets

   """

    basic_generator = ImageDataGenerator(rescale=1. / 255,
                                         validation_split=0.1)
    augmented_generator = ImageDataGenerator(
                                     rescale=1.0/255.0,
                                     featurewise_center= True,
                                     featurewise_std_normalization = True,
                                     rotation_range=15,
                                     width_shift_range=0.1,
                                     height_shift_range=0.1,
                                     zoom_range=0.2,
                                     brightness_range=[0.2,1.0],
                                     )

    train_augmented = augmented_generator.flow_from_dataframe(dataframe=train_data,
                                                    x_col="JPG",
                                                    y_col="Category",
                                                    color_mode="grayscale",
                                                    class_mode="categorical",
                                                    subset="training",
                                                    batch_size=20,
                                                    target_size=(200, 200))

    valid_augmented = basic_generator.flow_from_dataframe(dataframe=train_data,
                                                         x_col="JPG",
                                                         y_col="Category",
                                                         color_mode="grayscale",
                                                         class_mode="categorical",
                                                         subset="validation",
                                                         batch_size=20,
                                                         target_size=(200, 200))

    test_augmented = basic_generator.flow_from_dataframe(dataframe=test_data,
                                                   x_col="JPG",
                                                   y_col="Category",
                                                   color_mode="grayscale",
                                                   class_mode="categorical",
                                                   batch_size=20,
                                                   target_size=(200, 200))

    test_generator = ImageDataGenerator(rescale=1. / 255)

    pred = get_main_prediction_data(prediction_path)

    main_testing = test_generator.flow_from_dataframe(dataframe=pred,
                                                            x_col="JPG",
                                                            y_col=None,
                                                            color_mode="grayscale",
                                                            class_mode=None,
                                                            batch_size=20,
                                                            target_size=(200, 200))
    return train_augmented, valid_augmented, test_augmented, pred, main_testing

# Launch preprocessing

# plot_data(negative_data_path, positive_data_path)

train_data, test_data = split_data(negative_path, positive_path)

train, valid, test, eval, main_test = data_augmentation(train_data, test_data)

