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
TRAINED_PATH = 'models/nocs_train20230420T2348/mask_rcnn_nocs_train_0006.pth'

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

def model_with_weights(mode = 'Trained'):

    model = modellib.MaskRCNN(config=config, model_dir=MODEL_DIR)

    if mode == 'Trained':
        model.load_state_dict(torch.load(TRAINED_PATH))
    
    elif mode == 'COCO':

        pretrained_state_dict = torch.load(COCO_MODEL_PATH)

        filtered_state_dict = pretrained_state_dict

        mismatches = ["classifier.linear_class.weight","classifier.linear_class.bias","classifier.linear_bbox.weight","classifier.linear_bbox.bias","mask.conv5.weight","mask.conv5.bias"]

        for i in range(len(mismatches)):

            weights = filtered_state_dict[mismatches[i]]

            if weights.shape[0] == 81 and weights.dim() > 1:
                w1 = weights[[0,40,46]]
                w2 = weights[[64,42]]
                w3 = torch.zeros_like(w2)

                final_weights = torch.vstack((w1,w3,w2))
                pass

            # weights shape = (324,1024)
            # expected 28,1024
            # 0:3, 160:163, 184:187, 256:259, 168:171
            elif weights.shape[0] == 324 and len(weights.shape) > 1:
                weights = torch.reshape(weights, (81,4,1024))
                # weights = weights.view(weights.size()[0], -1, 4)
                w1 = weights[[0,40,46]]
                w2 = weights[[64,42]]
                w3 = torch.zeros_like(w2)

                final_weights = torch.vstack((w1.flatten(end_dim=-2),w3.flatten(end_dim=-2),w2.flatten(end_dim=-2)))

            elif weights.shape[0] == 324:
                weights = torch.reshape(weights, (81,4))
                w1 = weights[[0,40,46]]
                w2 = weights[[64,42]]
                w3 = torch.zeros_like(w2)

                final_weights = torch.cat((w1.flatten(),w3.flatten(),w2.flatten()))
            else:
                w1 = weights[[0,40,46]]
                w2 = weights[[64,42]]
                w3 = torch.zeros_like(w2)

                final_weights = torch.cat((w1,w3,w2))

            filtered_state_dict[mismatches[i]] = final_weights

        model.load_state_dict(filtered_state_dict, strict=False)

    return model

model = model_with_weights()

# Here we get all file names in image dir
file_names = [f for f in os.listdir(IMAGE_DIR) if f.endswith(( '.png'))]

# Decide between random choice or run on certain image
file_name = random.choice(file_names)
# file_name = file_names[5]

print(file_name)

image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name))

if image.shape[2] == 4:
    image = image[:,:,:3]

# Run detection
results = model.detect([image])

# Visualize results
r = results[0]

rois, masks, class_ids, scores, coords = r['rois'], r['masks'], r['class_ids'], r['scores'],r['coords']

visualize.plot_nocs(coords,file_name)

visualize.display_instances(image, rois, masks, class_ids, synset_names,file_name,scores)

plt.figure()
plt.imshow(masks.sum(2))
plt.savefig("output_images/mask_out.png")