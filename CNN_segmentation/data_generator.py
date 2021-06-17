import numpy as np
import nibabel as nib
import cv2
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import warnings
import preprocess_segmentation as ps
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ignore unnecessary warnings

warnings.simplefilter("ignore")

SIZE = 128

class DataGenerator(Sequence):

    """
       Class that handles data generation for keras

       Attributes
       ----------
       self : keras Sequence
           base object for fitting to a sequence of data, such as a dataset
       dim : array
           dimension of images
       batch_size : int
           size of the batches
       links : int
           number of links
       shuffle : boolean
           shuffling factor

       Methods
       -------
       on_epoch_end(self)
           Updates files after each epoch
    """

    def __init__(self, files, dim=(SIZE, SIZE), batch_size=1, links=2, shuffle=True):

        """
            Initialization

            Parameters
            ----------
            self : keras Sequence
                base object for fitting to a sequence of data, such as a dataset
            dim : array
                dimension of images
            batch_size : int
                size of the batches
            links : int
                number of links
            shuffle : boolean
                shuffling factor
        """
        self.files = files
        self.dim = dim
        self.batch_size = batch_size
        self.links = links
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):

        """
            Identify number of batches per epoch

            Parameters
            ----------
            self : keras Sequence
                base object for fitting to a sequence of data, such as a dataset

            Returns
            -------
            num_of_batches
                number of batches per epoch
        """
        num_of_batches = int(np.floor(len(self.files) / self.batch_size));
        return num_of_batches

    def __getitem__(self, index):
        """
            Generate a batch by index

            Parameters
            ----------
            self : keras Sequence
                base object for fitting to a sequence of data, such as a dataset
            index : int
                index of a single batch
        """
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        index_of_batch = [self.files[k] for k in indexes]
        x_1, x_2 = self.generation_of_data(index_of_batch)

        return x_1, x_2

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def generation_of_data(self, index_of_batch):

        """
           Generate data with set number and size of batches

           Parameters
           ----------
           self : keras Sequence
               base object for fitting to a sequence of data, such as a dataset
           index : int
               index of a single batch
        """
        x_1 = np.zeros((self.batch_size * ps.SCANS, *self.dim, self.links))
        x_2 = np.zeros((self.batch_size * ps.SCANS, 240, 240))
        x_3 = np.zeros((self.batch_size * ps.SCANS, *self.dim, 4))

        for c, i in enumerate(index_of_batch):
            case_path = os.path.join(ps.training_path, i)

            data_path = os.path.join(case_path, f'{i}_flair.nii');
            flair = nib.load(data_path).get_fdata()

            data_path = os.path.join(case_path, f'{i}_t1ce.nii');
            ce = nib.load(data_path).get_fdata()

            data_path = os.path.join(case_path, f'{i}_seg.nii');
            seg = nib.load(data_path).get_fdata()

            for j in range(ps.SCANS):
                x_1[j + ps.SCANS * c, :, :, 0] = cv2.resize(flair[:, :, j + ps.INITIAL], (SIZE, SIZE));
                x_1[j + ps.SCANS * c, :, :, 1] = cv2.resize(ce[:, :, j + ps.INITIAL], (SIZE, SIZE));
                x_2[j + ps.SCANS * c] = seg[:, :, j + ps.INITIAL];

        x_2[x_2 == 4] = 3;
        mask = tf.one_hot(x_2, 4);
        x_3 = tf.image.resize(mask, (SIZE, SIZE));
        return x_1 / np.max(x_1), x_3

    def get_files(self):
        return self.files;

    def get_dim(self):
        return self.dim;

    def get_batch_size(self):
        return self.batch_size;

    def get_links(self):
        return self.links;

    def get_shuffle(self):
        return self.shuffle;