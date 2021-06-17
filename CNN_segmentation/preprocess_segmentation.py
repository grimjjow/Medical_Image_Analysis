import nilearn as nl
import nilearn.plotting as nlplt
import matplotlib.pyplot as plt
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ignore unnecessary warnings

warnings.filterwarnings("ignore")

"""

This file contains implementation of functions for:
    - data retrieval
    - cropping and normalization
    - plotting data and ground truth
    - data split

"""

CATEGORIES = {
    0: 'HEALTHY',
    1: 'NCR/NET',
    2: 'ED',
    3: 'ET'
}

SCANS = 100
INITIAL = 26

training_path = '../CNN_segmentation/input/BraTS_2020_segment/MICCAI_BraTS_2020_TrainingData/'
validation_path = '../CNN_segmentation/input/BraTS_2020_segment/MICCAI_BraTS_2020_ValidationData/'

def plot_initial_data():

    """
      Method for plotting intial training data

      Parameters
      ----------
      training_path : os.path
          Path for training images
      validation_path : os.path
          Path for validation images

    """

    niimg = nl.image.load_img(training_path + 'BraTS20_Training_001/BraTS20_Training_001_flair.nii')
    nimask = nl.image.load_img(training_path + 'BraTS20_Training_001/BraTS20_Training_001_seg.nii')

    fig, axes = plt.subplots(nrows=4, figsize=(30, 40))

    nlplt.plot_anat(niimg,
                    title='BraTS20_Training_001_flair.nii plot_anat',
                    axes=axes[0])

    nlplt.plot_epi(niimg,
                   title='BraTS20_Training_001_flair.nii plot_epi',
                   axes=axes[1])

    nlplt.plot_img(niimg,
                   title='BraTS20_Training_001_flair.nii plot_img',
                   axes=axes[2])

    nlplt.plot_roi(nimask,
                   title='BraTS20_Training_001_flair.nii with mask plot_roi',
                   bg_img=niimg,
                   axes=axes[3], cmap='Paired')

    fig.savefig('../CNN_segmentation/plots_and_outputs/initial_data_sample.png')

# Launch preprocessing

# plot_initial_data()