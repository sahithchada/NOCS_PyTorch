import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

import coco
import utils
import model as modellib
import visualize
import glob
import torch
import cv2
from dataset import NOCSData
import datetime
import _pickle as cPickle
import time



# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "models/mask_rcnn_coco.pth")

TRAINED_PATH = 'models/NOCS_Trained_2.pth'

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


# Path to specific image
IMAGE_SPECIFIC = None

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # GPU_COUNT = 0 for CPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 6 # Background plus 6 classes

config = InferenceConfig()
config.display()

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


config.display()

model = modellib.MaskRCNN(config=config, model_dir=MODEL_DIR)
if config.GPU_COUNT==0:
    model.load_state_dict(torch.load(TRAINED_PATH,map_location=torch.device('cpu')))
else:
    model.load_state_dict(torch.load(TRAINED_PATH))
    device = torch.device('cuda')
    model.to(device)


save_dir = os.path.join('output')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
now = datetime.datetime.now()

# Whether to evaluate on synthetic or real data
use_camera_data=False

# Should be true to save detection results, set to false to evaluate
detect = False

# Whether to do Pose fitting
umeyama = True


if use_camera_data:
    camera_dir = os.path.join('data', 'camera')
    dataset = NOCSData(synset_names,'val')
    dataset.load_camera_scenes(camera_dir)
    dataset.prepare(class_map)

    data="camera/val"
    intrinsics = np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]]) #for camera data

else:
    gt_dir = os.path.join('data','gts','real_test')

    real_dir = os.path.join('data', 'real')
    dataset = NOCSData(synset_names,'test')
    dataset.load_real_scenes(real_dir)
    dataset.prepare(class_map)
    image_ids = dataset.image_ids

    data="real/test"
    intrinsics= np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]]) # for real data

if detect:
    for i, image_id in enumerate(image_ids):
        start_time = datetime.datetime.now()
        print('Image id: ', image_id)
        image_path = dataset.image_info[image_id]["path"]
        result = {}
        image = dataset.load_image(image_id)
        depth = dataset.load_depth(image_id)
        gt_mask, gt_coord, gt_class_ids, gt_scales, gt_domain_label = dataset.load_mask(image_id)
        gt_bbox = utils.extract_bboxes(gt_mask)
        result['image_id'] = image_id
        result['image_path'] = image_path
        result['gt_class_ids'] = gt_class_ids
        result['gt_bboxes'] = gt_bbox
        result['gt_RTs'] = None            
        result['gt_scales'] = gt_scales
        image_path_parsing = image_path.split('/')

        
        gt_pkl_path = os.path.join(gt_dir, 'results_{}_{}_{}.pkl'.format('real_test', image_path_parsing[-2], image_path_parsing[-1]))
        print(gt_pkl_path,image_id)
        if (os.path.exists(gt_pkl_path)):

                with open(gt_pkl_path, 'rb') as f:
                    gt = cPickle.load(f)
                result['gt_RTs'] = gt['gt_RTs']
                if 'handle_visibility' in gt:
                    result['gt_handle_visibility'] = gt['handle_visibility']
                    assert len(gt['handle_visibility']) == len(gt_class_ids)
                    print('got handle visibiity.')
                else: 
                    result['gt_handle_visibility'] = np.ones_like(gt_class_ids)
        else:
                # align gt coord with depth to get RT
                if not 'real_test' in ['coco_val', 'coco_train']:
                    #print('ooooooooh yeaaaaaaaaaaa')
                    if len(gt_class_ids) == 0:
                        print('No gt instance exsits in this image.')

                    print('\nAligning ground truth...')
                    start = time.time()
                    result['gt_RTs'], _, error_message, _ = utils.align(gt_class_ids, 
                                                                     gt_mask, 
                                                                     gt_coord, 
                                                                     depth, 
                                                                     intrinsics, 
                                                                     synset_names, 
                                                                     image_path,
                                                                     save_dir+'/'+'{}_{}_{}_gt_'.format(data, image_path_parsing[-2], image_path_parsing[-1]))
                    print('New alignment takes {:03f}s.'.format(time.time() - start))

                   

                result['gt_handle_visibility'] = np.ones_like(gt_class_ids)

        if image.shape[2] == 4:
           
            image = image[:,:,:3]  

        # Run detection
        with torch.no_grad():
            results = model.detect([image])
            r = results[0]
            rois, masks, class_ids, scores, coords = r['rois'], r['masks'], r['class_ids'], r['scores'],r['coords']

        r['coords'][:,:,:,2]=1-r['coords'][:,:,:,2]

        result['pred_class_ids'] = r['class_ids']
        result['pred_bboxes'] = r['rois']
        result['pred_RTs'] = None   
        result['pred_scores'] = r['scores']
        if len(r['class_ids']) == 0:
            print('No instance is detected.')

 
        if umeyama:
            result['pred_RTs'], result['pred_scales'], error_message, elapses =  utils.align(r['class_ids'], 
                                                                                    r['masks'], 
                                                                                    r['coords'], 
                                                                                    depth, 
                                                                                    intrinsics, 
                                                                                    synset_names,  image_path)
            draw_rgb=False
            result['gt_handle_visibility'] = np.ones_like(gt_class_ids)
            utils.draw_detections(image, save_dir, data, image_id, intrinsics, synset_names, draw_rgb,
                                                    gt_bbox, gt_class_ids, gt_mask, gt_coord, result['gt_RTs'], gt_scales, result['gt_handle_visibility'],
                                                    r['rois'], r['class_ids'], r['masks'], r['coords'], result['pred_RTs'], r['scores'], result['pred_scales'],draw_gt=True)
        end_time = datetime.datetime.now()
        
        path_parse = image_path.split('/')
        image_short_path = '_'.join(path_parse[-3:])
       
        save_path = os.path.join(save_dir, 'results_{}.pkl'.format(image_short_path))
        with open(save_path, 'wb') as f:
            cPickle.dump(result, f)

        print('Results of image {} has been saved to {}.'.format(image_short_path, save_path))
        execution_time = end_time - start_time
        print("Time taken for execution:", execution_time)

else:

    log_dir = "output/"

    result_pkl_list = glob.glob(os.path.join(log_dir, 'results_*.pkl'))
    result_pkl_list = sorted(result_pkl_list)
    assert len(result_pkl_list)
  


    final_results = []
    for pkl_path in result_pkl_list:
            with open(pkl_path, 'rb') as f:
                result = cPickle.load(f)
                if not 'gt_handle_visibility' in result:
                    result['gt_handle_visibility'] = np.ones_like(result['gt_class_ids'])
                    print('can\'t find gt_handle_visibility in the pkl.')
                else:
                    assert len(result['gt_handle_visibility']) == len(result['gt_class_ids']), "{} {}".format(result['gt_handle_visibility'], result['gt_class_ids'])


            if type(result) is list:
                final_results += result
            elif type(result) is dict:
                final_results.append(result)
            else:
                assert False
    
    aps = utils.compute_degree_cm_mAP(final_results, synset_names, log_dir,
                                                                    degree_thresholds = range(0, 61, 1),
                                                                    shift_thresholds= np.linspace(0, 1, 31)*15,  
                                                                    iou_3d_thresholds=np.linspace(0, 1, 101),
                                                                    iou_pose_thres=0.1,
                                                                    use_matches_for_pose=True)