import os
import time
import numpy as np
import skimage.io

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from skimage import exposure
import matplotlib.pyplot as plt
import visualize
import random

import zipfile
import urllib.request
import shutil

from config import Config
import utils
import model as modellib
# from dataset import TrainData

import torch
import argparse
import cv2

# Root directory of the project
ROOT_DIR = os.getcwd()

MODEL_DIR = os.path.join(ROOT_DIR, "models")

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "models/mask_rcnn_coco.pth")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = 2014
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

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
    # NUM_CLASSES = 1 + 80  # background + 6 object categories
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
    if COORD_USE_BINS:
         COORD_NUM_BINS = 32
    else:
        COORD_REGRESS_LOSS   = 'Soft_L1'
   
    COORD_SHARE_WEIGHTS = False
    COORD_USE_DELTA = False

    COORD_POOL_SIZE = 14
    COORD_SHAPE = [28, 28]

    USE_MINI_MASK = False

############################################################
#  Dataset
############################################################

class CocoDataset(utils.Dataset):
    def load_coco(self, dataset_dir, subset, year=DEFAULT_DATASET_YEAR, class_ids=None,
                  class_map=None, return_coco=False, auto_download=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        if auto_download is True:
            self.auto_download(dataset_dir, subset, year)

        coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco

    def auto_download(self, dataDir, dataType, dataYear):
        """Download the COCO dataset/annotations if requested.
        dataDir: The root directory of the COCO dataset.
        dataType: What to load (train, val, minival, valminusminival)
        dataYear: What dataset year to load (2014, 2017) as a string, not an integer
        Note:
            For 2014, use "train", "val", "minival", or "valminusminival"
            For 2017, only "train" and "val" annotations are available
        """

        # Setup paths and file names
        if dataType == "minival" or dataType == "valminusminival":
            imgDir = "{}/{}{}".format(dataDir, "val", dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, "val", dataYear)
            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format("val", dataYear)
        else:
            imgDir = "{}/{}{}".format(dataDir, dataType, dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, dataType, dataYear)
            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format(dataType, dataYear)
        # print("Image paths:"); print(imgDir); print(imgZipFile); print(imgURL)

        # Create main folder if it doesn't exist yet
        if not os.path.exists(dataDir):
            os.makedirs(dataDir)

        # Download images if not available locally
        if not os.path.exists(imgDir):
            os.makedirs(imgDir)
            print("Downloading images to " + imgZipFile + " ...")
            with urllib.request.urlopen(imgURL) as resp, open(imgZipFile, 'wb') as out:
                shutil.copyfileobj(resp, out)
            print("... done downloading.")
            print("Unzipping " + imgZipFile)
            with zipfile.ZipFile(imgZipFile, "r") as zip_ref:
                zip_ref.extractall(dataDir)
            print("... done unzipping")
        print("Will use images in " + imgDir)

        # Setup annotations data paths
        annDir = "{}/annotations".format(dataDir)
        if dataType == "minival":
            annZipFile = "{}/instances_minival2014.json.zip".format(dataDir)
            annFile = "{}/instances_minival2014.json".format(annDir)
            annURL = "https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0"
            unZipDir = annDir
        elif dataType == "valminusminival":
            annZipFile = "{}/instances_valminusminival2014.json.zip".format(dataDir)
            annFile = "{}/instances_valminusminival2014.json".format(annDir)
            annURL = "https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0"
            unZipDir = annDir
        else:
            annZipFile = "{}/annotations_trainval{}.zip".format(dataDir, dataYear)
            annFile = "{}/instances_{}{}.json".format(annDir, dataType, dataYear)
            annURL = "http://images.cocodataset.org/annotations/annotations_trainval{}.zip".format(dataYear)
            unZipDir = dataDir
        # print("Annotations paths:"); print(annDir); print(annFile); print(annZipFile); print(annURL)

        # Download annotations if not available locally
        if not os.path.exists(annDir):
            os.makedirs(annDir)
        if not os.path.exists(annFile):
            if not os.path.exists(annZipFile):
                print("Downloading zipped annotations to " + annZipFile + " ...")
                with urllib.request.urlopen(annURL) as resp, open(annZipFile, 'wb') as out:
                    shutil.copyfileobj(resp, out)
                print("... done downloading.")
            print("Unzipping " + annZipFile)
            with zipfile.ZipFile(annZipFile, "r") as zip_ref:
                zip_ref.extractall(unZipDir)
            print("... done unzipping")
        print("Will use annotations in " + annFile)

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

class SyntheticData(utils.Dataset):

    def __init__(self, synset_names, subset, config=Config()):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

        # which dataset: train/val/test
        self.subset = subset
        assert subset in ['train', 'val', 'test']

        self.config = config

        self.source_image_ids = {}

        # Add classes
        for i, obj_name in enumerate(synset_names):
            if i == 0:  ## class 0 is bg class
                continue
            self.add_class("BG", i, obj_name)  ## class id starts with 1

    def load_camera_scenes(self, dataset_dir, if_calculate_mean=False):
        """Load a subset of the CAMERA dataset.
        dataset_dir: The root directory of the CAMERA dataset.
        subset: What to load (train, val)
        if_calculate_mean: if calculate the mean color of the images in this dataset
        """

        image_dir = os.path.join(dataset_dir, self.subset)
        source = "CAMERA"
        num_images_before_load = len(self.image_info)

        folder_list = [name for name in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, name))]
        
        num_total_folders = len(folder_list)

        # image_ids = range(10*num_total_folders)
        color_mean = np.zeros((0, 3), dtype=np.float32)
        # Add images
        for folder_id in folder_list:

            folder_id = int(folder_id)

            for i in range(10):

                image_id = i

                image_path = os.path.join(image_dir, '{:05d}'.format(folder_id), '{:04d}'.format(image_id))
                color_path = image_path + '_color.png'
                if not os.path.exists(color_path):
                    continue
                
                meta_path = os.path.join(image_dir, '{:05d}'.format(folder_id), '{:04d}_meta.txt'.format(image_id))
                inst_dict = {}
                with open(meta_path, 'r') as f:
                    for line in f:
                        line_info = line.split(' ')
                        inst_id = int(line_info[0])  ##one-indexed
                        cls_id = int(line_info[1])  ##zero-indexed
                        # skip background objs
                        # symmetry_id = int(line_info[2])
                        inst_dict[inst_id] = cls_id

                width = self.config.IMAGE_MAX_DIM  # meta_data['viewport_size_x'].flatten()[0]
                height = self.config.IMAGE_MIN_DIM  # meta_data['viewport_size_y'].flatten()[0]

                self.add_image(
                    source=source,
                    image_id=image_id,
                    path=image_path,
                    width=width,
                    height=height,
                    inst_dict=inst_dict)

                if if_calculate_mean:
                    image_file = image_path + '_color.png'
                    image = cv2.imread(image_file).astype(np.float32)
                    print(i)
                    color_mean_image = np.mean(image, axis=(0, 1))[:3]
                    color_mean_image = np.expand_dims(color_mean_image, axis=0)
                    color_mean = np.append(color_mean, color_mean_image, axis=0)

        if if_calculate_mean:
            dataset_color_mean = np.mean(color_mean[::-1], axis=0)
            print('The mean color of this dataset is ', dataset_color_mean)

        num_images_after_load = len(self.image_info)
        self.source_image_ids[source] = np.arange(num_images_before_load, num_images_after_load)
        print('{} images are loaded into the dataset from {}.'.format(num_images_after_load - num_images_before_load, source))

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file.
        """
        info = self.image_info[image_id]
        if info["source"] in ["CAMERA", "Real"]:
            image_path = info["path"] + '_color.png'
            assert os.path.exists(image_path), "{} is missing".format(image_path)

            #depth_path = info["path"] + '_depth.png'
        elif info["source"]=='coco':
            image_path = info["path"]
        else:
            assert False, "[ Error ]: Unknown image source: {}".format(info["source"])

        # print(image_path)
        image = cv2.imread(image_path)[:, :, :3]
        image = image[:, :, ::-1]

        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image =  cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


        return image

    def load_depth(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file.
        """
        info = self.image_info[image_id]
        if info["source"] in ["CAMERA", "Real"]:
            depth_path = info["path"] + '_depth.png'
            depth = cv2.imread(depth_path, -1)

            if len(depth.shape) == 3:
                # This is encoded depth image, let's convert
                depth16 = np.uint16(depth[:, :, 1]*256) + np.uint16(depth[:, :, 2]) # NOTE: RGB is actually BGR in opencv
                depth16 = depth16.astype(np.uint16)
            elif len(depth.shape) == 2 and depth.dtype == 'uint16':
                depth16 = depth
            else:
                assert False, '[ Error ]: Unsupported depth type.'
        else:
            depth16 = None
            
        return depth16

    def image_reference(self, image_id):
        """Return the object data of the image."""
        info = self.image_info[image_id]
        if info["source"] in ["ShapeNetTOI", "Real"]:
            return info["inst_dict"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def process_data(self, mask_im, coord_map, inst_dict, meta_path, load_RT=False):
        # parsing mask
        cdata = mask_im
        cdata = np.array(cdata, dtype=np.int32)
        
        # instance ids
        instance_ids = list(np.unique(cdata))
        instance_ids = sorted(instance_ids)
        # remove background
        assert instance_ids[-1] == 255
        del instance_ids[-1]

        cdata[cdata==255] = -1
        assert(np.unique(cdata).shape[0] < 20)

        num_instance = len(instance_ids)
        h, w = cdata.shape

        # flip z axis of coord map
        coord_map = np.array(coord_map, dtype=np.float32) / 255
        coord_map[:, :, 2] = 1 - coord_map[:, :, 2]


        masks = np.zeros([h, w, num_instance], dtype=np.uint8)
        coords = np.zeros((h, w, num_instance, 3), dtype=np.float32)
        class_ids = np.zeros([num_instance], dtype=np.int_)
        scales = np.zeros([num_instance, 3], dtype=np.float32)

        with open(meta_path, 'r') as f:
            lines = f.readlines()

        scale_factor = np.zeros((len(lines), 3), dtype=np.float32)
        for i, line in enumerate(lines):
            words = line[:-1].split(' ')
            
            if len(words) == 3:
                ## real scanned objs
                if words[2][-3:] == 'npz':
                    npz_path = os.path.join(self.config.OBJ_MODEL_DIR, 'real_val', words[2])
                    with np.load(npz_path) as npz_file:
                        scale_factor[i, :] = npz_file['scale']
                else:
                    bbox_file = os.path.join(self.config.OBJ_MODEL_DIR, 'real_'+self.subset, words[2]+'.txt')
                    scale_factor[i, :] = np.loadtxt(bbox_file)

                scale_factor[i, :] /= np.linalg.norm(scale_factor[i, :])

            else:
                bbox_file = os.path.join(self.config.OBJ_MODEL_DIR, self.subset, words[2], words[3], 'bbox.txt')
                bbox = np.loadtxt(bbox_file)
                scale_factor[i, :] = bbox[0, :] - bbox[1, :]

        i = 0

        # delete ids of background objects and non-existing objects 
        inst_id_to_be_deleted = []
        for inst_id in inst_dict.keys():
            if inst_dict[inst_id] == 0 or (not inst_id in instance_ids):
                inst_id_to_be_deleted.append(inst_id)
        for delete_id in inst_id_to_be_deleted:
            del inst_dict[delete_id]


        for inst_id in instance_ids:  # instance mask is one-indexed
            if not inst_id in inst_dict:
                continue
            inst_mask = np.equal(cdata, inst_id)
            assert np.sum(inst_mask) > 0
            assert inst_dict[inst_id]
                
            masks[:, :, i] = inst_mask
            coords[:, :, i, :] = np.multiply(coord_map, np.expand_dims(inst_mask, axis=-1))

            # class ids is also one-indexed
            class_ids[i] = inst_dict[inst_id]
            scales[i, :] = scale_factor[inst_id - 1, :]
            i += 1

        # print('before: ', inst_dict)

        masks = masks[:, :, :i]
        coords = coords[:, :, :i, :]
        coords = np.clip(coords, 0, 1)

        class_ids = class_ids[:i]
        scales = scales[:i]

        return masks, coords, class_ids, scales

    def load_mask(self, image_id):
        """Generate instance masks for the objects in the image with the given ID.
        """
        info = self.image_info[image_id]
        #masks, coords, class_ids, scales, domain_label = None, None, None, None, None

        if info["source"] in ["CAMERA", "Real"]:
            domain_label = 0 ## has coordinate map loss

            mask_path = info["path"] + '_mask.png'
            coord_path = info["path"] + '_coord.png'

            assert os.path.exists(mask_path), "{} is missing".format(mask_path)
            assert os.path.exists(coord_path), "{} is missing".format(coord_path)

            inst_dict = info['inst_dict']
            meta_path = info["path"] + '_meta.txt'

            mask_im = cv2.imread(mask_path)[:, :, 2]
            coord_map = cv2.imread(coord_path)[:, :, :3]
            coord_map = coord_map[:, :, (2, 1, 0)]

            masks, coords, class_ids, scales = self.process_data(mask_im, coord_map, inst_dict, meta_path)

        else:
            assert False

        return masks, coords, class_ids, scales, domain_label
    
    def load_augment_data(self, image_id):
        """Generate augmented data for the image with the given ID.
        """
        info = self.image_info[image_id]
        image = self.load_image(image_id)

        # apply random gamma correction to the image
        gamma = np.random.uniform(0.8, 1)
        gain = np.random.uniform(0.8, 1)
        image = exposure.adjust_gamma(image, gamma, gain)

        # generate random rotation degree
        rotate_degree = np.random.uniform(-5, 5)

        if info["source"] in ["CAMERA", "Real"]:
            domain_label = 0 ## has coordinate map loss

            mask_path = info["path"] + '_mask.png'
            coord_path = info["path"] + '_coord.png'
            inst_dict = info['inst_dict']
            meta_path = info["path"] + '_meta.txt'

            mask_im = cv2.imread(mask_path)[:, :, 2]
            coord_map = cv2.imread(coord_path)[:, :, :3]
            coord_map = coord_map[:, :, ::-1]

            image, mask_im, coord_map = utils.rotate_and_crop_images(image, 
                                                                     masks=mask_im, 
                                                                     coords=coord_map, 
                                                                     rotate_degree=rotate_degree)
            masks, coords, class_ids, scales = self.process_data(mask_im, coord_map, inst_dict, meta_path)
        elif info["source"]=="coco":
            domain_label = 1 ## no coordinate map loss

            instance_masks = []
            class_ids = []
            annotations = self.image_info[image_id]["annotations"]
            # Build mask of shape [height, width, instance_count] and list
            # of class IDs that correspond to each channel of the mask.
            for annotation in annotations:
                class_id = self.map_source_class_id(
                    "coco.{}".format(annotation['category_id']))
                if class_id:
                    m = utils.annToMask(annotation, info["height"],
                                       info["width"])
                    # Some objects are so small that they're less than 1 pixel area
                    # and end up rounded out. Skip those objects.
                    if m.max() < 1:
                        continue
                    instance_masks.append(m)
                    class_ids.append(class_id)

            # Pack instance masks into an array
            masks = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)

            #print('\nbefore augmented, image shape: {}, masks shape: {}'.format(image.shape, masks.shape))
            image, masks = utils.rotate_and_crop_images(image, 
                                                        masks=masks, 
                                                        coords=None, 
                                                        rotate_degree=rotate_degree)
                        
            #print('\nafter augmented, image shape: {}, masks shape: {}'.format(image.shape, masks.shape))
            
            if len(masks.shape)==2:
                masks = masks[:, :, np.newaxis]
            
            final_masks = []
            final_class_ids = []
            for i in range(masks.shape[-1]):
                m = masks[:, :, i]
                if m.max() < 1:
                    continue
                final_masks.append(m)
                final_class_ids.append(class_ids[i])

            if final_class_ids:
                masks = np.stack(final_masks, axis=2)
                class_ids = np.array(final_class_ids, dtype=np.int32)
            else:
                # Call super class to return an empty mask
                masks = np.empty([0, 0, 0])
                class_ids = np.empty([0], np.int32)


            # use zero arrays as coord map for COCO images
            coords = np.zeros(masks.shape+(3,), dtype=np.float32)
            scales = np.ones((len(class_ids),3), dtype=np.float32)

        else:
            assert False


        return image, masks, coords, class_ids, scales, domain_label


#main training   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',  default='0', type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    print('Using GPU {}.'.format(args.gpu))

    config = Nocs_train_config()

    camera_train_dir = os.path.join('data', 'train')
    camera_val_dir=os.path.join('data', 'val')

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
    
    # synset_names = ['BG', #0
    #                 'bottle', #1
    #                 'bowl', #2
    #                 'laptop',#5
    #                 'mug'#6
    #                 ]


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
    # Load the state dictionary of the pre-trained model
    pretrained_state_dict = torch.load(COCO_MODEL_PATH)

    # List of layers to exclude, changed 
    # exclude_layers = ["classifier","mask"]

    # Filter out the layers to exclude from the state dictionary
    # filtered_state_dict = {k: v for k, v in pretrained_state_dict.items() if not any(layer in k for layer in exclude_layers)}
    filtered_state_dict = pretrained_state_dict

    mismatches = ["classifier.linear_class.weight","classifier.linear_class.bias","classifier.linear_bbox.weight","classifier.linear_bbox.bias","mask.conv5.weight","mask.conv5.bias"]


    """     
    classifier.linear_class.weight: og: ([81, 1024]) changed: torch.Size([7, 1024]).
    classifier.linear_class.bias: og: torch.Size([81]) changed: torch.Size([7]).
    classifier.linear_bbox.weight: og: torch.Size([324, 1024]) changed: torch.Size([28, 1024]).
    classifier.linear_bbox.bias: og: torch.Size([324]) changed: torch.Size([28]).
    mask.conv5.weight: og: torch.Size([81, 256, 1, 1]) changed: torch.Size([7, 256, 1, 1]).
    mask.conv5.bias: og: torch.Size([81]) changed: torch.Size([7]). 
    """

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

    # # Update the log directory
    model.set_log_dir(COCO_MODEL_PATH)

    if config.GPU_COUNT > 0:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print("Model to:",device)

    model.to(device)

    camera_dir = os.path.join('data', 'camera')
    # camera_dir = '../NOCS_CVPR2019/data/camera'

    trainset = SyntheticData(synset_names,'train')
    trainset.load_camera_scenes(camera_dir)
    trainset.prepare(class_map)

    valset = SyntheticData(synset_names,'val')
    valset.load_camera_scenes(camera_dir)
    valset.prepare(class_map)


    # Training - Stage 1
    print("Training network heads")
    model.train_model(trainset, valset,
                learning_rate=config.LEARNING_RATE,
                epochs=2,
                layers='heads')

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Training Resnet layer 4+")
    model.train_model(trainset, valset,
                learning_rate=config.LEARNING_RATE/10,
                epochs=4,
                layers='4+')


    # model = modellib.MaskRCNN(config=config, model_dir=MODEL_DIR)
    # model.to(device)
    # model.load_state_dict(torch.load('models/nocs_train20230420T2339/mask_rcnn_nocs_train_0006.pth'))
    
    # Training - Stage 3
    # Finetune layers from ResNet stage 3 and up
    print("Training Resnet layer 3+")
    model.train_model(trainset, valset,
                learning_rate=config.LEARNING_RATE/100,
                epochs=6,
                layers='all')
    



