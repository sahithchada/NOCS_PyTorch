import os
import torch
import pandas as pd
from torchvision.io import read_image
import numpy as np
import cv2
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import glob
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import math
from torchvision.transforms import InterpolationMode
from torchvision import utils


def process_data(mask_im, coord_map, inst_dict, meta_path):
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
        i += 1


    masks = masks[:, :, :i]
    coords = coords[:, :, :i, :]
    coords = np.clip(coords, 0, 1)

    class_ids = class_ids[:i]

    return masks, coords, class_ids


def load_mask(image_id,transform = None):
    """Generate instance masks for the objects in the image with the given ID.
    """

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


    cm2 = read_image(coord_path)[:, :, :3]
    mm2  = read_image(mask_path)[:,:,2]

    # print(mask_im.shape,mm2.shape)
    # print(cm2.shape,coord_map.shape)

    if transform:
        coord_map = torch.tensor(coord_map).unsqueeze(0).permute(0,3,1,2)[0]
        mask_im = torch.tensor(mask_im).unsqueeze(0).permute(0,1,2)

        coord_map = transform(coord_map)

        mask_im = transform(mask_im)

        coord_map = coord_map.numpy().transpose(1,2,0)
        mask_im = mask_im.numpy()[0]

    coord_map = coord_map[:, :, (2, 1, 0)]

    # masks, coords, class_ids = process_data(mask_im, coord_map, inst_dict, meta_path)

    return mask_im,coord_map,inst_dict

def custom_collate(batch):
    '''
    Adds extra keys to a batch
    '''
    

    batch_image = batch[0]['image'].unsqueeze(0)
    batch_mask = np.expand_dims(batch[0]['mask_im'],0)
    batch_coord = np.expand_dims(batch[0]['coord_map'],0)

    # print(type(batch_image),type(batch_mask),type(batch_coord))

    inst_dicts_list = []

    if len(batch) > 1:

        for i in range(1,len(batch)):

            batch_image = torch.cat((batch_image,batch[i]['image'].unsqueeze(0)))
            batch_mask = np.concatenate((batch_mask,np.expand_dims(batch[i]['mask_im'],0)))
            batch_coord = np.concatenate((batch_coord,np.expand_dims(batch[i]['coord_map'],0)))

            inst_dicts_list.append(batch[i]['inst_dict'])


    batch_return = {'image':batch_image,'mask_im':batch_mask,'coord_map':batch_coord,'inst_dict':inst_dicts_list}


    return batch_return


class TrainData(Dataset):
    def __init__(self, img_dir,transform = None):
        self.img_dir = img_dir
        self.img_annos = glob.glob(img_dir + '/**/*.txt', recursive = True)
        self.transform = transform

    def __len__(self):

        return len(self.img_annos)

    def __getitem__(self, idx):

        to_transform = torch.rand(1).item()

        img_path = self.img_annos[idx].split('_')[0]

        image = read_image(img_path+'_color.png')

        gamma = np.random.uniform(0.8, 1)
        gain = np.random.uniform(0.8, 1)

        image = TF.adjust_gamma(image,gamma,gain)
        
        if self.transform and to_transform > 0.5:
            transform = RotationCrop()
            image = transform(image,is_img = True)
            mask_im,coord_map,inst_dict = load_mask(img_path,transform=transform)
        else:
            mask_im,coord_map,inst_dict = load_mask(img_path)


        sample = {'image':image,'mask_im':mask_im,'coord_map':coord_map,'inst_dict':inst_dict}

        # sample = {'image':image,'mask_im':mask_im,'coord_map':coord_map}

        return sample
    
def overlap_nocs(image,coords):

    nocs_map = np.sum(coords,2)
    alpha = np.sum(nocs_map,2,keepdims = True)
    nocs_map_alpha = np.concatenate((nocs_map,alpha),axis=2)

    plt.imshow(TF.to_pil_image(image))
    plt.imshow(nocs_map_alpha)
    plt.pause(0.001)

def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        int(bb_w - 2 * x),
        int(bb_h - 2 * y)
    )

class RotationCrop:
    """Rotate by one of the given angles."""

    def __init__(self):
        self.angle = np.random.uniform(-5, 5)

    def __call__(self, x,is_img = False):

        if is_img:
            rotated_x = TF.rotate(x,self.angle,expand = True,interpolation = InterpolationMode.BILINEAR)

            rect = largest_rotated_rect(x.shape[-2], x.shape[-1], self.angle)

            cropped_x = TF.center_crop(rotated_x,rect)

            x_resized = TF.resize(cropped_x,(480,640),antialias = True,interpolation= InterpolationMode.BILINEAR)
        else:
            rotated_x = TF.rotate(x,self.angle,expand = True,interpolation = InterpolationMode.NEAREST)

            rect = largest_rotated_rect(x.shape[-2], x.shape[-1], self.angle)

            cropped_x = TF.center_crop(rotated_x,rect)

            x_resized = TF.resize(cropped_x,(480,640),antialias = True,interpolation = InterpolationMode.NEAREST)

        return x_resized
    
def show_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, nocs_batch = sample_batched['image'],sample_batched['coord_map']
    batch_size = len(images_batch)

    im_size = images_batch.size(2)

    grid_border_size = 2

    # print(nocs_batch.shape,images_batch.shape)
    nocs_batch = torch.tensor(nocs_batch).permute(0,3,1,2)

    alpha = (torch.sum(nocs_batch,1,keepdim=True) > 0.0) * 1.0
    print(nocs_batch.shape,alpha.shape)
    res = torch.hstack((nocs_batch,alpha))
    print(res.shape)

    grid = utils.make_grid(images_batch)
    grid2 = utils.make_grid(nocs_batch)
    # grid2 = utils.make_grid(res)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.imshow(grid2.numpy().transpose((1, 2, 0)))

    # for i in range(batch_size):
    #     plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
    #                 landmarks_batch[i, :, 1].numpy() + grid_border_size,
    #                 s=10, marker='.', c='r')

    #     plt.title('Batch from dataloader')


    
def main():

    
    trainset = TrainData('data/train',transform=True)

    # fig = plt.figure()

    # for i in range(len(trainset)):

    #     sample = trainset[i]

    #     ax = plt.subplot(1, 4, i + 1)
    #     plt.tight_layout()
    #     ax.set_title('Sample #{}'.format(i))
    #     ax.axis('off')
    #     overlap_nocs(sample['image'],sample['coords'])

    #     if i == 3:
    #         plt.show()
    #         break
    
    trainloader = DataLoader(trainset, batch_size=4, shuffle=False, num_workers=0,collate_fn=custom_collate)

    for i_batch, sample_batched in enumerate(trainloader):
        print(i_batch, sample_batched['image'].shape,
            sample_batched['coord_map'].shape)
 
    # observe 4th batch and stop.
        if i_batch == 3:
            plt.figure()
            show_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break

    


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
    