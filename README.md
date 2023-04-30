# PyTorch Implementation of "Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation"

## Introduction

This is a PyTorch implementation of [**Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation**](https://arxiv.org/pdf/1901.02970.pdf), a **CVPR 2019 oral** paper.
The original Tensorflow implementation can be found [here](https://github.com/hughw19/NOCS_CVPR2019). The framework is built on a PyTorch implemenation of Mask-RCNN, which can be found [here](https://github.com/multimodallearning/pytorch-mask-rcnn)

This repository includes:
* Source code of NOCS.
* Training code
* Detection and evaluation code
* Pre-trained weights

## Requirements
This code has been tested with
* CUDA 12.1 & 
* Python 3.10 & 3.8
* PyTorch 2.0

Replicate the conda environment using:
```
conda create --name <env> --file requirements.txt
```

## Implementation

This code implements the model of the original paper with the following settings:
* NOCS values treated as classification (bins)
* Unshared weights between NOCS heads
* Symmetric Loss
* Real & Synthetic data training (no COCO)


## Datasets
* CAMERA Dataset: [Training](http://download.cs.stanford.edu/orion/nocs/camera_train.zip)/[Test](http://download.cs.stanford.edu/orion/nocs/camera_val25K.zip)/[IKEA_backgrounds](http://download.cs.stanford.edu/orion/nocs/ikea_data.zip)/[Composed_depths](http://download.cs.stanford.edu/orion/nocs/camera_composed_depth.zip)
```diff
+ Composed depth images contain the depths of both foreground synthetic objects and background real scenes for all train and validation data
```
* Real Dataset: [Training](http://download.cs.stanford.edu/orion/nocs/real_train.zip)/[Test](http://download.cs.stanford.edu/orion/nocs/real_test.zip)
* Ground truth pose annotation (for an easier evaluation): [Val&Real_test](http://download.cs.stanford.edu/orion/nocs/gts.zip)
* [Object Meshes](http://download.cs.stanford.edu/orion/nocs/obj_models.zip)

You can download the files and store them under data/. The data folder general structure is shown:
```bash
.
└── data/
    ├── camera/
    │   ├── train
    │   └── val
    ├── real/
    │   ├── train
    │   └── test
    ├── obj_models/
    │   ├── real_test
    │   ├── real_train
    │   ├── train
    │   └── val
    ├── camera_full_depths/
    │   ├── train
    │   └── val
    ├── gts/
    │   ├── real_test
    │   └── val
    └── ikea_data
```

## Pretrained weights
You can find the following checkpoints in this [download link](https://drive.google.com/uc?export=download&id=1SeNduFmmuFugT-1SE186YEPahM61JrAH):
* NOCS RCNN jointly trained on CAMERA, Real & MS COCO with 32 bin classification setting (Two sets of weights)
* Mask RCNN pretrained on MS COCO dataset

You can download the checkpoints and store them under models/.

## Training
```
# Train a new model from pretrained COCO weights
python train.py
```

## Detection and evaluation



