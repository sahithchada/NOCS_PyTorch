import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import coco
import utils
import model as modellib
import visualize

import torch
import cv2
from dataset import SyntheticData

from train import model_loaded_weights


# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
# Download this file and place in the models directory
# download link- https://drive.google.com/drive/folders/1LXUgC2IZUYNEoXr05tdqyKFZY0pZyPDc
# project (See README file for details)

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "models/mask_rcnn_coco.pth")
# TRAINED_PATH = 'models\mask_rcnn_nocs_train_0010.pth'
TRAINED_PATH = 'models/nocs_train20230422T1230/mask_rcnn_nocs_train_0025.pth'

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


# Path to specific image
# IMAGE_SPECIFIC = 'images/0000_color.png'
IMAGE_SPECIFIC = None

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # GPU_COUNT = 0 for CPU
    GPU_COUNT = 0
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 6 # Background plus 6 classes

config = InferenceConfig()
config.display()

#  real classes
coco_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush']

synset_names = ['BG', #0
                    'bottle', #1
                    'bowl', #2
                    'camera', #3
                    'can',  #4
                    'laptop',#5
                    'mug'#6
                    ]


class_map = {
        'bottle': 'bottle',
        'bowl':'bowl',
        'cup':'mug',
        'laptop': 'laptop',
    }


coco_cls_ids = []
for coco_cls in class_map:
    ind = coco_names.index(coco_cls)
    coco_cls_ids.append(ind)
config.display()

model = model_loaded_weights(config,inference=True,trained_path=TRAINED_PATH)


def run_model(model,fl_path = None ):

    if fl_path:
        image = skimage.io.imread(fl_path)
        file_name = fl_path.split('_')[-2].split('/')[-1]
        print(file_name+'.png')

    else:
        
        # Here we get all file names in image dir
        file_names = [f for f in os.listdir(IMAGE_DIR) if f.endswith(( '.png'))]

        # Decide between random choice or run on certain image
        file_name = random.choice(file_names)
        print(file_name)

        image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name))

    if image.shape[2] == 4:
        image = image[:,:,:3]  

    # Run detection
    results = model.detect([image])

    # Visualize results
    r = results[0]

    rois, masks, class_ids, scores, coords = r['rois'], r['masks'], r['class_ids'], r['scores'],r['coords']

    visualize.plot_nocs(coords,masks,file_name)

    visualize.display_instances(image, rois, masks, class_ids, synset_names,file_name,scores)
    return r



r=run_model(model,fl_path=IMAGE_SPECIFIC)

dataset = SyntheticData(synset_names,'train')
dataset.load_camera_scenes(camera_dir)
dataset.prepare(class_map)
image = dataset.load_image(image_id)

intrinsics = np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]]) #for camera data
result = {}
umeyama=True
if umeyama:
    result['pred_RTs'], result['pred_scales'], error_message, elapses =  utils.align(r['class_ids'], 
                                                                                        r['masks'], 
                                                                                        r['coords'], 
                                                                                        depth, 
                                                                                        intrinsics, 
                                                                                        synset_names,  image_path)
