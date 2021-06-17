from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import preprocess_detection
import warnings

warnings.filterwarnings("ignore")

"""

This file contains implementation of function for:
    - visual prediction 
    
"""

def make_prediction():

    model = load_model('medical_detection_cnn_model_v1.h5')

    prediction = model.predict(preprocess_detection.main_test)

    prediction = prediction.argmax(axis=-1)

    categories = []
    [categories.append('Negative') if i == 1 else categories.append('Positive') for i in prediction]

    fig, axes = plt.subplots(nrows=4,
                             ncols=4,
                             figsize=(20, 20),
                            subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(plt.imread(preprocess_detection.eval["JPG"].loc[i]))
        ax.set_title(f"Tumor prediction:{categories[i]}")
    plt.tight_layout()
    plt.show()
    fig.savefig('../CNN_detection/plots_and_outputs/detection_prediction.png')

# Launch prediction

make_prediction()

