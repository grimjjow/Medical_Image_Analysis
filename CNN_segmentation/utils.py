import tensorflow.keras.backend as K
import warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ignore unnecessary warnings

warnings.simplefilter("ignore")

"""

This file contains implementation of functions for:
    - calculate performance metrics

"""

def dice_coef(g_truth, pred, smooth=1):

    """
       Method to calculate dice coef for a whole tumor

       Parameters
       ----------
       g_truth :
            ground truth
       pred :
            prediction

       Returns
       -------
       sum
           dice coef value
    """
    num_of_categories = 4

    for i in range(num_of_categories):
        gt = K.flatten(g_truth[:, :, :, i])
        pr = K.flatten(pred[:, :, :, i])
        intersection = K.sum(gt * pr)
        loss = ((2. * intersection + smooth) / (K.sum(gt) + K.sum(pr) + smooth))
        if i == 0:
            sum = loss
        else:
            sum = sum + loss
    sum = sum / num_of_categories
    return sum


def dice_coef_nct(g_truth, pred, epsilon=1e-6):

    """
       Method to calculate dice coef for a necrotic tumoral region

       Parameters
       ----------
       g_truth :
            ground truth
       pred :
            prediction

       Returns
       -------
       dice coef value of NCT
    """
    intersection = K.sum(K.abs(g_truth[:, :, :, 1] * pred[:, :, :, 1]))
    return (2. * intersection) / (K.sum(K.square(g_truth[:, :, :, 1])) + K.sum(K.square(pred[:, :, :, 1])) + epsilon)


def dice_coef_ed(g_truth, pred, epsilon=1e-6):

    """
       Method to calculate dice coef for an edema tumoral region

       Parameters
       ----------
       g_truth :
            ground truth
       pred :
            prediction

       Returns
       -------
       dice coef value of ED
    """
    intersection = K.sum(K.abs(g_truth[:, :, :, 2] * pred[:, :, :, 2]))
    return (2. * intersection) / (K.sum(K.square(g_truth[:, :, :, 2])) + K.sum(K.square(pred[:, :, :, 2])) + epsilon)


def dice_coef_et(g_truth, pred, epsilon=1e-6):

    """
       Method to calculate dice coef for an enchancing tumoral region

       Parameters
       ----------
       g_truth :
            ground truth
       pred :
            prediction

       Returns
       -------
       dice coef value of ET
    """
    intersection = K.sum(K.abs(g_truth[:, :, :, 3] * pred[:, :, :, 3]))
    return (2. * intersection) / (K.sum(K.square(g_truth[:, :, :, 3])) + K.sum(K.square(pred[:, :, :, 3])) + epsilon)


def precision(g_truth, pred):

    """
       Method to calculate precision

       Parameters
       ----------
       g_truth :
            ground truth
       pred :
            prediction

       Returns
       -------
       precision value
    """
    true_positives = K.sum(K.round(K.clip(g_truth * pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def sensitivity(g_truth, pred):

    """
       Method to calculate recall(sensitivity)

       Parameters
       ----------
       g_truth :
            ground truth
       pred :
            prediction

       Returns
       -------
       recall value
    """
    true_positives = K.sum(K.round(K.clip(g_truth * pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(g_truth, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


