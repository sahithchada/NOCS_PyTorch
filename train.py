# Main training file
# Note that pre-trained weights are loaded from models directory
# Config may be altered

# Imports
import os
import numpy as np
from config import Config
import model as modellib
import torch
import argparse
from dataset import NOCSData, CocoDataset

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to save model folders
MODEL_DIR = os.path.join(ROOT_DIR, "models")

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "models/mask_rcnn_coco.pth")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    GPU_COUNT = 0
    # We use one GPU with 8GB memory, which can fit one image.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 16

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes

class Nocs_train_config(Config):
    # config file for nocs training, derives from base config  
    NAME="NOCS_train"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 6  # background + 6 object categories
    MEAN_PIXEL = np.array([[ 120.66209412, 114.70348358, 105.81269836]])

    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 640

    RPN_ANCHOR_SCALES = (16, 32, 48, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 64

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50

    WEIGHT_DECAY = 0.0001
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    COORD_LOSS_SCALE = 1
    
    COORD_USE_BINS = True
    COORD_NUM_BINS = 32
   
    COORD_SHARE_WEIGHTS = False
    COORD_USE_DELTA = False

    COORD_POOL_SIZE = 14
    COORD_SHAPE = [28, 28]

    USE_MINI_MASK = False

def model_loaded_weights(config, rand_weights = False, trained_path = None):

    """     
    Loading MaskRCNN model with additional heads using config file.
    trained_path : path to NOCS trained weights in pth format
    If none, we presume training on the classes in synset and weights are chosen accordingly  
    """
    
    model = modellib.MaskRCNN(config=config, model_dir=MODEL_DIR)

    # Load the state dictionary of the pre-trained model
    pretrained_state_dict = torch.load(COCO_MODEL_PATH)

    if trained_path:

        model.load_state_dict(torch.load(trained_path))

    elif rand_weights:           

        # List of layers to exclude, changed 
        exclude_layers = ["classifier","mask"]

        # Filter out the layers to exclude from the state dictionary
        filtered_state_dict = {k: v for k, v in pretrained_state_dict.items() if not any(layer in k for layer in exclude_layers)}

        model.load_state_dict(filtered_state_dict, strict=False)

    else:
        filtered_state_dict = pretrained_state_dict

        mismatches = ["classifier.linear_class.weight","classifier.linear_class.bias","classifier.linear_bbox.weight","classifier.linear_bbox.bias","mask.conv5.weight","mask.conv5.bias"]


        # """     
        # classifier.linear_class.weight: og: ([81, 1024]) changed: torch.Size([7, 1024]).
        # classifier.linear_class.bias: og: torch.Size([81]) changed: torch.Size([7]).
        # classifier.linear_bbox.weight: og: torch.Size([324, 1024]) changed: torch.Size([28, 1024]).
        # classifier.linear_bbox.bias: og: torch.Size([324]) changed: torch.Size([28]).
        # mask.conv5.weight: og: torch.Size([81, 256, 1, 1]) changed: torch.Size([7, 256, 1, 1]).
        # mask.conv5.bias: og: torch.Size([81]) changed: torch.Size([7]). 
        # """

        for i in range(len(mismatches)):

            weights = filtered_state_dict[mismatches[i]]

            if weights.shape[0] == 81 and weights.dim() > 1:
                w1 = weights[[0,40,46]]
                w2 = weights[[64,42]]
                w3 = torch.zeros_like(w2)

                final_weights = torch.vstack((w1,w3,w2))
                pass

            elif weights.shape[0] == 324 and len(weights.shape) > 1:
                weights = torch.reshape(weights, (81,4,1024))

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


    model.set_log_dir(COCO_MODEL_PATH)

    if config.GPU_COUNT > 0 and torch.cuda.is_available():
        device = torch.device('cuda')
        
    else:
        device = torch.device('cpu')

    print("Model to:", device)

    model.to(device)

    return model

#main training   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',  default='0', type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    

    config = Nocs_train_config()

    # Defining camera and real directories
    camera_dir = os.path.join('data', 'camera')
    real_dir = os.path.join('data', 'real')


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
    

    # 0, 40, 46, rand, rand, 64, 42
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

    # trained_path = None if training a new model
    # Default mode assumes 7 classes: BG, Bottle, bowl, camera, can, laptop, mug
    # Use rand_weights = True if not using the default classes
    model = model_loaded_weights(config, rand_weights = False, trained_path=None)
        
    # Load and prep synthetic train data
    synthtrain = NOCSData(synset_names,'train')
    synthtrain.load_camera_scenes(camera_dir)
    synthtrain.prepare(class_map)

    # Load and prep real train data
    realtrain = NOCSData(synset_names,'train')
    realtrain.load_real_scenes(real_dir)
    realtrain.prepare(class_map)

    # Load and prep synthetic validation data
    valset = NOCSData(synset_names,'val')
    valset.load_camera_scenes(camera_dir)
    valset.prepare(class_map)


    # Training - Stage 1
    print("Training network heads")
    model.train_model([synthtrain,realtrain], valset,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads')

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Training Resnet layer 4+")
    model.train_model([synthtrain,realtrain], valset,
                learning_rate=config.LEARNING_RATE/10,
                epochs=5,
                layers='4+')


    
    # Training - Stage 3
    # Finetune layers from ResNet stage 3 and up
    print("Training Resnet layer 3+")
    model.train_model([synthtrain,realtrain], valset,
                learning_rate=config.LEARNING_RATE/100,
                epochs=70,
                layers='all')