import os
import torch
import pandas as pd
from torchvision.io import read_image
import numpy as np
import cv2
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import glob
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader


def process_data(mask_im, coord_map, inst_dict, meta_path, load_RT=False):
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
    # coord_map[:, :, 2] = 1 - coord_map[:, :, 2]


    masks = np.zeros([h, w, num_instance], dtype=np.uint8)
    coords = np.zeros((h, w, num_instance, 3), dtype=np.float32)
    class_ids = np.zeros([num_instance], dtype=np.int_)
    scales = np.zeros([num_instance, 3], dtype=np.float32)

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
        i += 1


    masks = masks[:, :, :i]
    coords = coords[:, :, :i, :]
    coords = np.clip(coords, 0, 1)

    class_ids = class_ids[:i]

    return masks, coords, class_ids


def load_mask(image_id):
    """Generate instance masks for the objects in the image with the given ID.
    """
    # info = self.image_info[image_id]

    # general_path = 'data/train/00012' + '/' + image_id

    general_path = image_id

    mask_path = general_path + '_mask.png'
    coord_path = general_path + '_coord.png'
    meta_path = general_path + '_meta.txt'

    assert os.path.exists(mask_path), "{} is missing".format(mask_path)
    assert os.path.exists(coord_path), "{} is missing".format(coord_path)

    inst_dict = {}
    with open(meta_path, 'r') as f:
        for line in f:
            line_info = line.split(' ')
            inst_id = int(line_info[0])  ##one-indexed
            cls_id = int(line_info[1])  ##zero-indexed
            # skip background objs
            # symmetry_id = int(line_info[2])
            inst_dict[inst_id] = cls_id

    mask_im = cv2.imread(mask_path)[:, :, 2]
    coord_map = cv2.imread(coord_path)[:, :, :3]
    coord_map = coord_map[:, :, (2, 1, 0)]

    masks, coords, class_ids = process_data(mask_im, coord_map, inst_dict, meta_path)

    return masks, coords, class_ids


class TrainData(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_annos = glob.glob(img_dir + '/**/*.txt', recursive = True)

    def __len__(self):

        return len(self.img_annos)

    def __getitem__(self, idx):

        dumbo = self.img_annos[idx]
        dumbo = dumbo.split('_')[0]

        img_path = dumbo

        image = read_image(img_path+'_color.png')
        
        masks, coords, class_ids = load_mask(img_path)

        sample = {'image':image,'masks':masks,'coords':coords,'class_ids':class_ids}

        return sample
    
def overlap_nocs(image,coords):

    # mask_binary = np.max(masks,2)
    nocs_map = np.sum(coords,2)
    alpha = np.sum(nocs_map,2,keepdims = True)
    nocs_map_alpha = np.concatenate((nocs_map,alpha),axis=2)

    # print(nocs_map.shape,alpha.shape)

    plt.imshow(F.to_pil_image(image))
    plt.imshow(nocs_map_alpha)
    plt.pause(0.001)


    
def main():

    trainset = TrainData('data/train')

    fig = plt.figure()

    for i in range(len(trainset)):

        sample = trainset[i]

        # print(i, sample['image'].shape, sample['masks'].shape)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        overlap_nocs(sample['image'],sample['coords'])

        if i == 3:
            plt.show()
            break
    
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True)


    # while(1):

    #     img_id = input("Enter img_id:")
    #     masks, coords, class_ids = load_mask(img_id)
    #     print(masks.shape,coords.shape,class_ids.shape)


    #     mask_binary = np.max(masks,2)
    #     nocs_map = np.sum(coords,2)
    #     nocs_map[nocs_map == 0.] = 1.0

    #     plt.figure()
    #     plt.subplot(2,1,1)
    #     plt.imshow(mask_binary,cmap='binary')

    #     plt.subplot(2,1,2)
    #     plt.imshow(nocs_map,cmap='brg')

    #     plt.show()

if __name__ == "__main__":
    main()
    