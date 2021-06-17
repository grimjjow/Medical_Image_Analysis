from tensorflow.keras.models import load_model
import preprocess_detection
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt_False_Positive_vs_True_Positive
import warnings
warnings.filterwarnings("ignore")

"""

This file contains implementation of functions for:
    - evaluation of the detection model
    - calculate performance metrics
    
"""

def evaluate_cnn_model():

    """
        Method for testing of the model

    """

    model = load_model('medical_detection_cnn_model_v1.h5')
    test_set = preprocess_detection.test
    score = model.evaluate(test_set, verbose=0)
    print('\n', 'Test loss:', score[0])
    print('\n', 'Test accuracy:', score[1])

def plot_metrics(model):

    """
       Method for calculating the performance metrics

    """

    test_set = preprocess_detection.test
    categories = test_set.classes;
    pred = model.predict_generator(test_set, steps=len(test_set), verbose=0)
    prob = pred[:, 1]
    binary = prob > 0.5

    print('\nConfusion Matrix\n')
    print(confusion_matrix(categories, binary));
    accuracy = accuracy_score(categories, binary)
    print('Accuracy: %f' % accuracy)
    precision = precision_score(categories, binary)
    print('Precision: %f' % precision)
    recall = recall_score(categories, binary)
    print('Recall: %f' % recall)
    f1 = f1_score(categories, binary)
    print('F1 score: %f' % f1)
    auc = roc_auc_score(categories, prob)
    print('ROC AUC: %f' % auc)

evaluate_cnn_model()