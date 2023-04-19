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


# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
# Download this file and place in the models directory
# download link- https://drive.google.com/drive/folders/1LXUgC2IZUYNEoXr05tdqyKFZY0pZyPDc
# project (See README file for details)

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "models/mask_rcnn_coco.pth")
MODEL_WEIGHTS_PATH = os.path.join(ROOT_DIR, "models/nocs_train20230418T0205/mask_rcnn_nocs_train_0010.pth")
TRAINED_PATH = 'models/nocs_train20230419T0004/mask_rcnn_nocs_train_0050.pth'

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # GPU_COUNT = 0 for CPU
    GPU_COUNT = 0
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 6 # Background plus 6 classes

config = InferenceConfig()
config.display()

# Create model object.
model = modellib.MaskRCNN(model_dir=MODEL_DIR, config=config)
if config.GPU_COUNT:
    model = model.cuda()

# # Load weights trained on MS-COCO
# pretrained_weights = torch.load(COCO_MODEL_PATH)
# model_state_dict = model.state_dict()
# for key in model_state_dict.keys():
#     if key in pretrained_weights:
#         # If the layer exists in pretrained weights, copy the weights
#         model_state_dict[key] = pretrained_weights[key]
#     else:
#         # If the layer does not exist in pretrained weights, initialize with random weights
#         shape = model_state_dict[key].shape
#         model_state_dict[key] = torch.randn(shape).normal_(mean=0, std=3.0)

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

# Create an instance of your PyTorch MaskRCNN model with the given configuration
model = modellib.MaskRCNN(config=config, model_dir=MODEL_DIR)


model.load_state_dict(torch.load(TRAINED_PATH))
# Load the state dictionary of the pre-trained model
# pretrained_state_dict = torch.load(COCO_MODEL_PATH)
# # List of layers to exclude, changed 
# exclude_layers = ["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask","mask","classifier"]
# # Filter out the layers to exclude from the state dictionary
# filtered_state_dict = {k: v for k, v in pretrained_state_dict.items() if not any(layer in k for layer in exclude_layers)}
# # Load the modified state dictionary into your model
# model.load_state_dict(filtered_state_dict, strict=False)

# # Load the updated state dictionary into the model
# model.load_state_dict(model_state_dict)

#model.load_state_dict(torch.load(COCO_MODEL_PATH))


# Here we decide to use coco imgs or pngs from NOCS data
file_names = [f for f in os.listdir(IMAGE_DIR) if f.endswith(( '.png'))]
# file_names = [f for f in os.listdir(IMAGE_DIR) if f.endswith(( '.jpg'))]


# Decide between random choice or run on certain image
file_name = random.choice(file_names)
# file_name = file_names[0]

print(file_name)

image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name))

# Run detection
results = model.detect([image])

# Visualize results
r = results[0]

rois, masks, class_ids, scores, coords = r['rois'], r['masks'], r['class_ids'], r['scores'],r['coords']

visualize.plot_nocs(coords,file_name)

visualize.display_instances(image, rois, masks, class_ids, synset_names,file_name,scores)


# plt.savefig("output_images/output.png")
# plt.show()