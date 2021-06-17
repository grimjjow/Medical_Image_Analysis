import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from warnings import filterwarnings
import cv2
import tensorflow as tf
from tensorflow.keras.models import *
import utils
import train_segmentation as ts

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ignore unnecessary warnings

filterwarnings("ignore", category=DeprecationWarning)
filterwarnings("ignore", category=FutureWarning)
filterwarnings("ignore", category=UserWarning)

"""

This file contains implementation of function for:
    - visual prediction 

"""

SIZE = 128

CATEGORIES = {
    0: 'HEALTHY',
    1: 'NCR/NET',
    2: 'ED',
    3: 'ET'
}

SCANS = 100
INITIAL = 26

model = load_model('../CNN_segmentation/medical_segmentation_cnn_model.h5',
                                   custom_objects={ 'accuracy' : tf.keras.metrics.MeanIoU(num_classes=4),
                                                   "dice_coef": utils.dice_coef,
                                                   "precision": utils.precision,
                                                   "sensitivity":utils.sensitivity,
                                                   "dice_coef_necrotic": utils.dice_coef_nct,
                                                   "dice_coef_edema": utils.dice_coef_ed,
                                                   "dice_coef_enhancing": utils.dice_coef_et
                                                  }, compile=False)

def path_pred(case_path, case):

    """
      Method for connecting the path to prediction images

      Parameters
      ----------
      case_path : os.path
          Path for prediction
      case : images in .nii format
          Specific patient case

    """
    path = next(os.walk(case_path))[2]
    x_1 = np.empty((SCANS, SIZE, SIZE, 2))

    vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_flair.nii');
    flair = nib.load(vol_path).get_fdata()

    vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_t1ce.nii');
    ce = nib.load(vol_path).get_fdata()

    for j in range(SCANS):
        x_1[j, :, :, 0] = cv2.resize(flair[:, :, j + INITIAL], (SIZE, SIZE))
        x_1[j, :, :, 1] = cv2.resize(ce[:, :, j + INITIAL], (SIZE, SIZE))
    return model.predict(x_1 / np.max(x_1), verbose=1)


def visual_path_pred(case, initial_scan = 55):

    """
     Method for connecting the path to prediction images

     Parameters
     ----------
     case : images in .nii format
         Specific patient case
     initial_scan : int
         Value of intial scan slice


   """
    path = f"../CNN_segmentation/input/BraTS_2020_segment/MICCAI_BraTS_2020_TrainingData/BraTS20_Training_{case}"
    gt = nib.load(os.path.join(path, f'BraTS20_Training_{case}_seg.nii')).get_fdata()
    initial = nib.load(os.path.join(path, f'BraTS20_Training_{case}_flair.nii')).get_fdata()
    p = path_pred(path, case)

    core = p[:, :, :, 1]
    edema = p[:, :, :, 2]
    enhancing = p[:, :, :, 3]

    plt.figure(figsize=(18, 50))

    f, axarr = plt.subplots(1, 6, figsize=(18, 50))

    for i in range(6):
        axarr[i].imshow(cv2.resize(initial[:, :, initial_scan + INITIAL], (SIZE, SIZE)), cmap="gray",
                        interpolation='none')

    axarr[0].imshow(cv2.resize(initial[:, :, initial_scan + INITIAL], (SIZE, SIZE)), cmap="gray")
    axarr[0].title.set_text('Initial image flair')
    curr_gt = cv2.resize(gt[:, :, initial_scan + INITIAL], (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    axarr[1].imshow(curr_gt, cmap="Reds", interpolation='none', alpha=0.3)
    axarr[1].title.set_text('Ground truth')
    axarr[2].imshow(p[initial_scan, :, :, 1:4], cmap="Reds", interpolation='none', alpha=0.3)
    axarr[2].title.set_text('Whole tumor')
    axarr[3].imshow(edema[initial_scan, :, :], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[3].title.set_text(f'{CATEGORIES[1]} tumor')
    axarr[4].imshow(core[initial_scan, :, ], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[4].title.set_text(f'{CATEGORIES[2]} tumor')
    axarr[5].imshow(enhancing[initial_scan, :, ], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[5].title.set_text(f'{CATEGORIES[3]} tumor')
    plt.show()
    # f.savefig('../CNN_segmentation/plots_and_outputs/segmentation_prediction.png')

# Launch visual prediction

visual_path_pred(case=ts.test_ids[0][-3:])
visual_path_pred(case=ts.test_ids[1][-3:])
visual_path_pred(case=ts.test_ids[2][-3:])
visual_path_pred(case=ts.test_ids[3][-3:])
visual_path_pred(case=ts.test_ids[4][-3:])
visual_path_pred(case=ts.test_ids[5][-3:])
visual_path_pred(case=ts.test_ids[6][-3:])
visual_path_pred(case=ts.test_ids[7][-3:])
visual_path_pred(case=ts.test_ids[8][-3:])

