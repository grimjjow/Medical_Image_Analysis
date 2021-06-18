# Medical_Image_Analysis
# Deep Learning Techniques for Detection and Segmentation of Brain Tumors

## Data

OASIS dataset for detection is avaliable [here](https://www.oasis-brains.org).  
BraTS dataset for segmentation is avaliable [here](https://www.oasis-brains.org).  

## Prerequisites

1. Download data from the links provided and adjust the paths.

2. Install the following packages:
* cv2
* glob
* keras
* matplotlib
* nibabel
* numpy
* pandas
* os
* warnings

3. Check the python version: python 3.9

## Preprocsessing

For detection run the following:

```
$ python3 preprocess_detection.py
```

For segmentation run the following:

```
$ python3 preprocess_segmentation.py
```

## Training

You can use our [pretrained models](https://drive.google.com/file/d/16Q7EYKp_lTXPVr8P6kH9EMZ-EKC9CFPb/view?usp=sharing) to get the same results as in our paper.
Or you can run the training for detection using: 

```
$ python3 train_detection.py
```
and for segmentation using:

```
$ python3 train_segmentation.py
```

## Testing

Detection model can be evaluated by using:

```
$ python3 test_detection.py
```
and segmentation model can be evaluated by using:

```
$ python3 test_segmentation.py
```

## Prediction

To obtain some visual results for both models run:

```
$ python3 predict_detection.py
```
or

```
$ python3 predict_segmentation.py
```

## References

This work in based on some state-of-the-art approaches, which are listed below.

1. Image augmentation for machine learning experiments https://github.com/aleju/imgaug
2. Image Segmentation Keras https://github.com/divamgupta/image-segmentation-keras 
3. DeepSeg https://github.com/razeineldin/DeepSeg
4. Capstone Project: Automating retinal disease detection with convolutional neural network https://github.com/trankhavy/capstone_segmentation
5. Unet Brain tumor segmentation https://github.com/naomifridman/Unet_Brain_tumor_segmentation
6. Medical Image Diagnosis https://github.com/YuanningEric/Medical_Image_Diagnosis
