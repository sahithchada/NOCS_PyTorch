"""
Mask R-CNN
Common utility functions and classes.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import sys
import os
import math
import random
import numpy as np
import scipy.misc
import scipy.ndimage
import skimage.color
import skimage.io
import torch
from PIL import Image
from skimage.transform import resize
import cv2

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.ops import RoIAlign
import time
import matplotlib.pyplot as plt

from aligning import estimateSimilarityTransform

############################################################
#  Bounding Boxes
############################################################

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps

def box_refinement(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]
    """

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = torch.log(gt_height / height)
    dw = torch.log(gt_width / width)

    result = torch.stack([dy, dx, dh, dw], dim=1)
    return result


############################################################
#  Dataset
############################################################

class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.

        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.d
        """
        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        #self.num_classes = len(self.class_info)
        self.num_classes = 0

        #self.class_ids = np.arange(self.num_classes)
        self.class_ids = []

        #self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.class_names = []


        #self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
        #                              for info, id in zip(self.class_info, self.class_ids)}
        self.class_from_source_map = {}

        for cls_info in self.class_info:
            source = cls_info["source"]
            if source == 'coco':
                map_key = "{}.{}".format(cls_info['source'], cls_info['id'])
                self.class_from_source_map[map_key] = self.class_names.index(class_map[cls_info["name"]])
            else:
                self.class_ids.append(self.num_classes)
                self.num_classes += 1
                self.class_names.append(cls_info["name"])

                map_key = "{}.{}".format(cls_info['source'], cls_info['id'])
                self.class_from_source_map[map_key] = self.class_ids[-1]


        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)


        # Mapping from source class and image IDs to internal IDs
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))


        '''
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)
        '''

        print(self.class_names)
        print(self.class_from_source_map)
        print(self.sources)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id] if source_class_id in self.class_from_source_map else None

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    def append_data(self, class_info, image_info):
        self.external_to_class_id = {}
        for i, c in enumerate(self.class_info):
            for ds, id in c["map"]:
                self.external_to_class_id[ds + str(id)] = i

        # Map external image IDs to internal ones.
        self.external_to_image_id = {}
        for i, info in enumerate(self.image_info):
            self.external_to_image_id[info["ds"] + str(info["id"])] = i

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's availble online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        return image

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids


def resize_image(image, min_dim=None, max_dim=None, padding=False):
    """
    Resizes an image keeping the aspect ratio.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    padding: If true, pads image with zeros so it's size is max_dim x max_dim

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    # Resize image and mask
    if scale != 1:
        height, width = image.shape[:2]
        resized_height, resized_width = round(height * scale), round(width * scale)
        resized_image = Image.fromarray(image).resize((resized_width, resized_height))
        image=np.array(resized_image)

    # Need padding?
    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding


# def resize_mask(mask, scale, padding):
#     """Resizes a mask using the given scale and padding.
#     Typically, you get the scale and padding from resize_image() to
#     ensure both, the image and the mask, are resized consistently.

#     scale: mask scaling factor
#     padding: Padding to add to the mask in the form
#             [(top, bottom), (left, right), (0, 0)]
#     """
#     print(mask.shape)
#     h, w = mask.shape[:2]
#     mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
#     mask = np.pad(mask, padding, mode='constant', constant_values=0)
#     return mask

def resize_mask(mask, scale, padding):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image, the mask, and the coordinate map are resized consistently.
    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    h, w = mask.shape[:2]
    # for instance mask
    if len(mask.shape) == 3:
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
        new_padding = padding
    # for coordinate map
    elif len(mask.shape) == 4:
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1, 1], order=0)
        new_padding = padding + [(0, 0)]
    else:
        assert False

    mask = np.pad(mask, new_padding, mode='constant', constant_values=0)

    return mask


def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to cut memory load.
    Mini-masks can then resized back to image scale using expand_masks()

    See inspect_data.ipynb notebook for more details.
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        #m = scipy.misc.imresize(m.astype(float), mini_shape, interp='bilinear')
        #change
        #m = resize(m.astype(float), mini_shape, mode='reflect', order=1, anti_aliasing=False)
        m = Image.fromarray(m)
        m = m.resize((mini_shape), resample=Image.BILINEAR)
        m= np.array(m, dtype=np.float32)
        mini_mask[:, :, i] = np.where(m >= 128, 1, 0)
    return mini_mask


def expand_mask(bbox, mini_mask, image_shape):
    """Resizes mini masks back to image size. Reverses the change
    of minimize_mask().

    See inspect_data.ipynb notebook for more details.
    """
    mask = np.zeros(image_shape[:2] + (mini_mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mini_mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        h = y2 - y1
        w = x2 - x1
        m = scipy.misc.imresize(m.astype(float), (h, w), interp='bilinear')
        mask[y1:y2, x1:x2, i] = np.where(m >= 128, 1, 0)
    return mask


# TODO: Build and use this function to reduce code duplication
def mold_mask(mask, config):
    pass


def unmold_mask(mask, bbox, image_shape):
    """Converts a mask generated by the neural network into a format similar
    to its original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.
    Returns a binary mask with the same size as the original image.
    """
    threshold = 0.5
    y1, x1, y2, x2 = bbox
    mask = Image.fromarray(mask)
    mask = mask.resize((x2 - x1, y2 - y1), resample=Image.BILINEAR)
    mask = np.array(mask, dtype=np.float32)
    mask = np.where(mask >= threshold, 1, 0).astype(np.uint8)

    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask

def unmold_coord(coord, bbox, image_shape):
    """Converts a mask generated by the neural network into a format similar
    to it's original shape.
    coord: [height, width, 3] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.
    Returns a coordinate map with the same size as the original image.
    """
    y1, x1, y2, x2 = bbox

    #max_coord_x = np.amax(coord[:, :, 0])
    #max_coord_y = np.amax(coord[:, :, 1])
    #max_coord_z = np.amax(coord[:, :, 2])

    #print('before resize:')
    #print(max_coord_x, max_coord_y, max_coord_z)

    #coord = scipy.misc.imresize(
    #    coord, (y2 - y1, x2 - x1, 3), interp='nearest').astype(np.float32)/ 255.0
    #    #coord, (y2 - y1, x2 - x1, 3), interp='bilinear').astype(np.uint8)
    coord = cv2.resize(coord, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)

    #max_coord_x_resize = np.amax(coord[:, :, 0])
    #max_coord_y_resize = np.amax(coord[:, :, 1])
    #max_coord_z_resize = np.amax(coord[:, :, 2])

    #print('after resize:')
    #print(max_coord_x_resize, max_coord_y_resize, max_coord_z_resize)


    # Put the mask in the right location.
    full_coord= np.zeros(image_shape, dtype=np.float32)
    full_coord[y1:y2, x1:x2, :] = coord
    return full_coord

############################################################
#  Anchors
############################################################

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)


############################################################
#  IMAGE AUGMENTATION
############################################################

def calculate_rotation(image_size, angle):
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    return new_w, new_h, affine_mat


def rotate_image(image, new_w, new_h, affine_mat, interpolation=cv2.INTER_LINEAR):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=interpolation
    )

    return result


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
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]

def rotate_and_crop(image, rotate_degree, interpolation):
    image_height, image_width = image.shape[0:2]


    new_w, new_h, affine_mat = calculate_rotation(image.shape[0:2][::-1], rotate_degree)
    image_rotated = rotate_image(image, new_w, new_h, affine_mat, interpolation)
    image_rotated_cropped = crop_around_center(
        image_rotated,
        *largest_rotated_rect(
            image_width,
            image_height,
            math.radians(rotate_degree)
        )
    )

    return image_rotated_cropped


def rotate_and_crop_images(image, masks, coords, rotate_degree):

    image_height, image_width = image.shape[0:2]
    new_w, new_h, affine_mat = calculate_rotation(image.shape[0:2][::-1], rotate_degree)

    image_rotated = rotate_image(image, new_w, new_h, affine_mat, cv2.INTER_LINEAR)
    mask_rotated = rotate_image(masks, new_w, new_h, affine_mat, cv2.INTER_NEAREST)
    
    rect = largest_rotated_rect(
            image_width,
            image_height,
            math.radians(rotate_degree)
        )

    image_rotated_cropped = crop_around_center(image_rotated, *rect)
    mask_rotated_cropped = crop_around_center(mask_rotated, *rect)

    image_rotated_cropped = cv2.resize(image_rotated_cropped, (image_width, image_height),interpolation=cv2.INTER_LINEAR)
    mask_rotated_cropped = cv2.resize(mask_rotated_cropped, (image_width, image_height), interpolation=cv2.INTER_NEAREST)

    if coords is not None:
        coord_rotated = rotate_image(coords, new_w, new_h, affine_mat, cv2.INTER_NEAREST)
        coord_rotated_cropped = crop_around_center(coord_rotated, *rect)
        coord_rotated_cropped = cv2.resize(coord_rotated_cropped, (image_width, image_height), interpolation=cv2.INTER_NEAREST)

        return image_rotated_cropped, mask_rotated_cropped, coord_rotated_cropped
    else:
        return image_rotated_cropped, mask_rotated_cropped

############################################################
#  Torch Helpers
############################################################

def log2(x):
    """Implementatin of Log2. Pytorch doesn't have a native implemenation."""
    ln2 = Variable(torch.log(torch.FloatTensor([2.0])), requires_grad=False)
    if x.is_cuda:
        ln2 = ln2.cuda()
    return torch.log(x + 1e-5) / ln2

class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__
    
############################################################
#  ROIAlign Layer
############################################################

def pyramid_roi_align(inputs, pool_size, image_shape):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_size: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, channels]. Shape of input image in pixels

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, channels, height, width]

    Output:
    Pooled regions in the shape: [num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    # Currently only supports batchsize 1
    for i in range(len(inputs)):
        inputs[i] = inputs[i].squeeze(0)

    # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
    boxes = inputs[0]
    # Feature Maps. List of feature maps from different level of the
    # feature pyramid. Each is [batch, height, width, channels]
    feature_maps = inputs[1:]
    #print(feature_maps[3].shape)

    # Assign each ROI to a level in the pyramid based on the ROI area.
    y1, x1, y2, x2 = boxes.chunk(4, dim=1)
    h = y2 - y1
    w = x2 - x1

    # Equation 1 in the Feature Pyramid Networks paper. Account for
    # the fact that our coordinates are normalized here.
    # e.g. a 224x224 ROI (in pixels) maps to P4
    image_area = Variable(torch.FloatTensor([float(image_shape[0]*image_shape[1])]), requires_grad=False)
    if boxes.is_cuda:
        image_area = image_area.cuda()
    roi_level = 4 + log2(torch.sqrt(h*w)/(224.0/torch.sqrt(image_area)))
    roi_level = roi_level.round().int()
    roi_level = roi_level.clamp(2,5)


    # Loop through levels and apply ROI pooling to each. P2 to P5.
    pooled = []
    box_to_level = []
    for i, level in enumerate(range(2, 6)):
        ix  = roi_level==level
        if not ix.any():
            continue
        ix = torch.nonzero(ix)[:,0]
        level_boxes = boxes[ix.data, :]

        # Keep track of which box is mapped to which level
        box_to_level.append(ix.data)

        # Stop gradient propogation to ROI proposals
        level_boxes = level_boxes.detach()

        # Crop and Resize
        # From Mask R-CNN paper: "We sample four regular locations, so
        # that we can evaluate either max or average pooling. In fact,
        # interpolating only a single value at each bin center (without
        # pooling) is nearly as effective."
        #
        # Here we use the simplified approach of a single value per bin,
        # which is how it's done in tf.crop_and_resize()
        # Result: [batch * num_boxes, pool_height, pool_width, channels]
        ind = Variable(torch.zeros(level_boxes.size()[0]),requires_grad=False).int()
        level_boxes = level_boxes[:, [1, 0, 3, 2]]
        n,h,w=feature_maps[i].shape
        level_boxes[:,[0, 2]]*= image_shape[0]
        level_boxes[:,[1,3]]*=image_shape[1]
        indexes = torch.zeros(level_boxes.shape[0], 1)
        if level_boxes.is_cuda:
            indexes = indexes.cuda()
        level_boxes = torch.cat((indexes, level_boxes), dim=1)
        if level_boxes.is_cuda:
            ind = ind.cuda()

        feature_maps_reshaped = torch.reshape(feature_maps[i], (1,n, h,w))

        roi_align1 = RoIAlign((pool_size, pool_size), spatial_scale=feature_maps[i].shape[1]/image_shape[0],sampling_ratio=-1)

        pooled_features=roi_align1(feature_maps_reshaped,level_boxes)
        pooled.append(pooled_features)

    # Pack pooled features into one tensor
    pooled = torch.cat(pooled, dim=0)

    # Pack box_to_level mapping into one array and add another
    # column representing the order of pooled boxes
    box_to_level = torch.cat(box_to_level, dim=0)

    # Rearrange pooled features to match the order of the original boxes
    _, box_to_level = torch.sort(box_to_level)
    pooled = pooled[box_to_level, :, :]

    return pooled

## functions for umeyama


def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.
    x: [rows, columns].
    """

    pre_shape = x.shape
    assert len(x.shape) == 2, x.shape
    new_x = x[~np.all(x == 0, axis=1)]
    post_shape = new_x.shape
    assert pre_shape[0] == post_shape[0]
    assert pre_shape[1] == post_shape[1]

    return new_x

def backproject(depth, intrinsics, instance_mask):
    intrinsics_inv = np.linalg.inv(intrinsics)
    image_shape = depth.shape
    width = image_shape[1]
    height = image_shape[0]

    x = np.arange(width)
    y = np.arange(height)

    #non_zero_mask = np.logical_and(depth > 0, depth < 5000)
    non_zero_mask = (depth > 0)
    final_instance_mask = np.logical_and(instance_mask, non_zero_mask)
    
    idxs = np.where(final_instance_mask)
    grid = np.array([idxs[1], idxs[0]])

    # shape: height * width
    # mesh_grid = np.meshgrid(x, y) #[height, width, 2]
    # mesh_grid = np.reshape(mesh_grid, [2, -1])
    length = grid.shape[1]
    ones = np.ones([1, length])
    uv_grid = np.concatenate((grid, ones), axis=0) # [3, num_pixel]

    xyz = intrinsics_inv @ uv_grid # [3, num_pixel]
    xyz = np.transpose(xyz) #[num_pixel, 3]

    z = depth[idxs[0], idxs[1]]

    # print(np.amax(z), np.amin(z))
    pts = xyz * z[:, np.newaxis]/xyz[:, -1:]
    pts[:, 0] = -pts[:, 0]
    pts[:, 1] = -pts[:, 1]

    return pts, idxs

def align(class_ids, masks, coords, depth, intrinsics, synset_names, image_path, save_path=None, if_norm=False, with_scale=True, verbose=False):
    num_instances = len(class_ids)
    error_messages = ''
    elapses = []
    if num_instances == 0:
        return np.zeros((0, 4, 4)), np.ones((0, 3)), error_messages, elapses

    RTs = np.zeros((num_instances, 4, 4))
    bbox_scales = np.ones((num_instances, 3))
    
    for i in range(num_instances):
        #class_name = synset_names[class_ids[i]]
        class_id = class_ids[i]
        mask = masks[:, :, i]
        coord = coords[:, :, i, :]
        abs_coord_pts = np.abs(coord[mask==1] - 0.5)
        bbox_scales[i, :] = 2*np.amax(abs_coord_pts, axis=0)

        pts, idxs = backproject(depth, intrinsics, mask)
        coord_pts = coord[idxs[0], idxs[1], :] - 0.5

        if if_norm:
            scale = np.linalg.norm(bbox_scales[i, :])
            bbox_scales[i, :] /= scale
            coord_pts /= scale

        
        try:
            start = time.time()
            
            scales, rotation, translation, outtransform = estimateSimilarityTransform(coord_pts, pts, False)

            aligned_RT = np.zeros((4, 4), dtype=np.float32) 
            if with_scale:
                aligned_RT[:3, :3] = np.diag(scales) / 1000 @ rotation.transpose()
            else:
                aligned_RT[:3, :3] = rotation.transpose()
            aligned_RT[:3, 3] = translation / 1000
            aligned_RT[3, 3] = 1
            
            if save_path is not None:
                coord_pts_rotated = aligned_RT[:3, :3] @ coord_pts.transpose() + aligned_RT[:3, 3:]
                coord_pts_rotated = coord_pts_rotated.transpose()
                np.savetxt(save_path+'_{}_{}_depth_pts.txt'.format(i, class_id), pts)
                np.savetxt(save_path+'_{}_{}_coord_pts.txt'.format(i, class_id), coord_pts)
                np.savetxt(save_path+'_{}_{}_coord_pts_aligned.txt'.format(i, class_id), coord_pts_rotated)

            if verbose:
                print('Mask ID: ', i)
                print('Scale: ', scales/1000)
                print('Rotation: ', rotation.transpose())
                print('Translation: ', translation/1000)

            elapsed = time.time() - start
            print('elapsed: ', elapsed)
            elapses.append(elapsed)
        

        except Exception as e:
            message = '[ Error ] aligning instance {} in {} fails. Message: {}.'.format(synset_names[class_id], image_path, str(e))
            print(message)
            error_messages += message + '\n'
            aligned_RT = np.identity(4, dtype=np.float32) 

        # print('Estimation takes {:03f}s.'.format(time.time() - start))
        # from camera world to computer vision frame
        z_180_RT = np.zeros((4, 4), dtype=np.float32)
        z_180_RT[:3, :3] = np.diag([-1, -1, 1])
        z_180_RT[3, 3] = 1

        RTs[i, :, :] = z_180_RT @ aligned_RT 

    return RTs, bbox_scales, error_messages, elapses

def compute_overlaps_masks(masks1, masks2):
    '''Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    '''
    # flatten masks
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps

def compute_RT_degree_cm_symmetry(RT_1, RT_2, class_id, handle_visibility, synset_names):
    '''
    :param RT_1: [4, 4]. homogeneous affine transformation
    :param RT_2: [4, 4]. homogeneous affine transformation
    :return: theta: angle difference of R in degree, shift: l2 difference of T in centimeter


    synset_names = ['BG',  # 0
                    'bottle',  # 1
                    'bowl',  # 2
                    'camera',  # 3
                    'can',  # 4
                    'cap',  # 5
                    'phone',  # 6
                    'monitor',  # 7
                    'laptop',  # 8
                    'mug'  # 9
                    ]
    
    synset_names = ['BG',  # 0
                    'bottle',  # 1
                    'bowl',  # 2
                    'camera',  # 3
                    'can',  # 4
                    'laptop',  # 5
                    'mug'  # 6
                    ]
    '''

    ## make sure the last row is [0, 0, 0, 1]
    if RT_1 is None or RT_2 is None:
        return -1
    try:
        assert np.array_equal(RT_1[3, :], RT_2[3, :])
        assert np.array_equal(RT_1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(RT_1[3, :], RT_2[3, :])
        exit()

    R1 = RT_1[:3, :3] / np.cbrt(np.linalg.det(RT_1[:3, :3]))
    T1 = RT_1[:3, 3]

    R2 = RT_2[:3, :3] / np.cbrt(np.linalg.det(RT_2[:3, :3]))
    T2 = RT_2[:3, 3]

#     try:
#         assert np.abs(np.linalg.det(R1) - 1) < 0.01
#         assert np.abs(np.linalg.det(R2) - 1) < 0.01
#     except AssertionError:
#         print(np.linalg.det(R1), np.linalg.det(R2))

    if synset_names[class_id] in ['bottle', 'can', 'bowl']:  ## symmetric when rotating around y-axis
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    elif synset_names[class_id] == 'mug' and handle_visibility==0:  ## symmetric when rotating around y-axis
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    elif synset_names[class_id] in ['phone', 'eggbox', 'glue']:
        y_180_RT = np.diag([-1.0, 1.0, -1.0])
        R = R1 @ R2.transpose()
        R_rot = R1 @ y_180_RT @ R2.transpose()
        theta = min(np.arccos((np.trace(R) - 1) / 2),
                    np.arccos((np.trace(R_rot) - 1) / 2))
    else:
        R = R1 @ R2.transpose()
        theta = np.arccos((np.trace(R) - 1) / 2)

    theta *= 180 / np.pi
    shift = np.linalg.norm(T1 - T2) * 100
    result = np.array([theta, shift])

    return result

def draw(img, imgpts, axes, color):
    imgpts = np.int32(imgpts).reshape(-1, 2)


    # draw ground layer in darker color
    color_ground = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
    for i, j in zip([4, 5, 6, 7],[5, 7, 4, 6]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_ground, 3)


    # draw pillars in blue color
    color_pillar = (int(color[0]*0.6), int(color[1]*0.6), int(color[2]*0.6))
    for i, j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_pillar, 3)

    
    # finally, draw top layer in color
    for i, j in zip([0, 1, 2, 3],[1, 3, 0, 2]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color, 3)


    # draw axes
    img = cv2.line(img, tuple(axes[0]), tuple(axes[1]), (0, 0, 255), 3)
    img = cv2.line(img, tuple(axes[0]), tuple(axes[3]), (255, 0, 0), 3)
    img = cv2.line(img, tuple(axes[0]), tuple(axes[2]), (0, 255, 0), 3) ## y last


    return img

def compute_matches(gt_boxes, gt_class_ids, gt_masks,
                    pred_boxes, pred_class_ids, pred_scores, pred_masks,
                    iou_threshold=0.5, score_threshold=0.0):
    """Finds matches between prediction and ground truth instances.
    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    if len(gt_class_ids)==0 or len(pred_class_ids) == 0:
        return -1 * np.ones([len(gt_class_ids)]), -1 * np.ones([len(pred_class_ids)]), None, np.zeros([0])

    pre_len = len(gt_boxes)
    gt_boxes = trim_zeros(gt_boxes)
    after_len = len(gt_boxes)
    assert pre_len == after_len
    gt_masks = gt_masks[..., :gt_boxes.shape[0]]

    pre_len = len(pred_boxes)
    pred_boxes = trim_zeros(pred_boxes)
    after_len = len(pred_boxes)
    assert pre_len == after_len
    pred_scores = pred_scores[:pred_boxes.shape[0]]

    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    
    
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break

    return gt_match, pred_match, overlaps, indices


def get_3d_bbox(scale, shift = 0):
    """
    Input: 
        scale: [3] or scalar
        shift: [3] or scalar
    Return 
        bbox_3d: [3, N]

    """
    if hasattr(scale, "__iter__"):
        bbox_3d = np.array([[scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2]]) + shift
    else:
        bbox_3d = np.array([[scale / 2, +scale / 2, scale / 2],
                  [scale / 2, +scale / 2, -scale / 2],
                  [-scale / 2, +scale / 2, scale / 2],
                  [-scale / 2, +scale / 2, -scale / 2],
                  [+scale / 2, -scale / 2, scale / 2],
                  [+scale / 2, -scale / 2, -scale / 2],
                  [-scale / 2, -scale / 2, scale / 2],
                  [-scale / 2, -scale / 2, -scale / 2]]) +shift

    bbox_3d = bbox_3d.transpose()
    return bbox_3d

def transform_coordinates_3d(coordinates, RT):
    """
    Input: 
        coordinates: [3, N]
        RT: [4, 4]
    Return 
        new_coordinates: [3, N]

    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = RT @ coordinates
    new_coordinates = new_coordinates[:3, :]/new_coordinates[3, :]
    return new_coordinates


def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Input: 
        coordinates: [3, N]
        intrinsics: [3, 3]
    Return 
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates

def draw_text(draw_image, bbox, text, draw_box=False):
    fontFace = cv2.FONT_HERSHEY_TRIPLEX
    fontScale = 1
    thickness = 1
    

    retval, baseline = cv2.getTextSize(text, fontFace, fontScale, thickness)
    
    bbox_margin = 10
    text_margin = 10
    
    text_box_pos_tl = (min(bbox[1] + bbox_margin, 635 - retval[0] - 2* text_margin) , min(bbox[2] + bbox_margin, 475 - retval[1] - 2* text_margin)) 
    text_box_pos_br = (text_box_pos_tl[0] + retval[0] + 2* text_margin,  text_box_pos_tl[1] + retval[1] + 2* text_margin)

    # text_pose is the bottom-left corner of the text
    text_pos = (text_box_pos_tl[0] + text_margin, text_box_pos_br[1] - text_margin - 3)
    
    if draw_box:
        cv2.rectangle(draw_image, 
                      (bbox[1], bbox[0]),
                      (bbox[3], bbox[2]),
                      (255, 0, 0), 2)

    cv2.rectangle(draw_image, 
                  text_box_pos_tl,
                  text_box_pos_br,
                  (255,0,0), -1)
    
    cv2.rectangle(draw_image, 
                  text_box_pos_tl,
                  text_box_pos_br,
                  (0,0,0), 1)

    cv2.putText(draw_image, text, text_pos,
                fontFace, fontScale, (255,255,255), thickness)

    return draw_image

def draw_detections(image, save_dir, data_name, image_id, intrinsics, synset_names, draw_rgb_coord,
                    gt_bbox, gt_class_ids, gt_mask, gt_coord, gt_RTs, gt_scales, gt_handle_visibility,
                    pred_bbox, pred_class_ids, pred_mask, pred_coord, pred_RTs, pred_scores, pred_scales,
                    draw_gt=True, draw_pred=True, draw_tag=False):

    alpha = 0.5

    if draw_gt:
        output_path = os.path.join(save_dir, '{}_{}_coord_gt.png'.format(data_name, image_id))
        draw_image = image.copy()
        num_gt_instances = len(gt_class_ids)

        for i in range(num_gt_instances):
            mask = gt_mask[:, :, i]
            #mask = mask[:, :, np.newaxis]
            #mask = np.repeat(mask, 3, axis=-1)
            cind, rind = np.where(mask == 1)
            coord_data = gt_coord[:, :, i, :].copy()
            coord_data[:, :, 2] = 1 - coord_data[:, :, 2] # undo the z axis flipping to match original data        
            draw_image[cind, rind] = coord_data[cind, rind] * 255
            
        if draw_tag:
            for i in range(num_gt_instances):
                overlay = draw_image.copy()
                overlay = draw_text(overlay, gt_bbox[i], synset_names[gt_class_ids[i]], draw_box=True)
                cv2.addWeighted(overlay, alpha, draw_image, 1 - alpha, 0, draw_image)
                   
        cv2.imwrite(output_path, draw_image[:, :, ::-1])

        plt.figure()
        plt.imshow(draw_image)
        plt.savefig('hi.png')


        output_path = os.path.join(save_dir, '{}_{}_bbox_gt.png'.format(data_name, image_id))
        draw_image_bbox = image.copy()

        if gt_RTs is not None:
            for ind, RT in enumerate(gt_RTs):
                cls_id = gt_class_ids[ind]

                xyz_axis = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
                transformed_axes = transform_coordinates_3d(xyz_axis, RT)
                projected_axes = calculate_2d_projections(transformed_axes, intrinsics)


                bbox_3d = get_3d_bbox(gt_scales[ind], 0)
                transformed_bbox_3d = transform_coordinates_3d(bbox_3d, RT)
                projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
                draw_image_bbox = draw(draw_image_bbox, projected_bbox, projected_axes, (255, 0, 0))

        cv2.imwrite(output_path, draw_image_bbox[:, :, ::-1])

        plt.figure()
        plt.imshow(draw_image_bbox[:, :, ::-1])
        plt.savefig('hi.png')


    if draw_pred:
        # Vs, Fs = dataset.load_objs(image_id, is_normalized=True) ## scale is estimated in RT
        output_path   = os.path.join(save_dir, '{}_{}_coord_pred.png'.format(data_name, image_id))
        output_path_r = os.path.join(save_dir, '{}_{}_coord_pred_r.png'.format(data_name, image_id))
        output_path_g = os.path.join(save_dir, '{}_{}_coord_pred_g.png'.format(data_name, image_id))
        output_path_b = os.path.join(save_dir, '{}_{}_coord_pred_b.png'.format(data_name, image_id))
        # utils.draw_coord_mask(image, r['class_ids'], pred_RTs, Vs, Fs, intrinsics, output_path)
        draw_image = image.copy()
        if draw_rgb_coord:
            r_image = image.copy()
            g_image = image.copy()
            b_image = image.copy()

        
        num_pred_instances = len(pred_class_ids)    
        for i in range(num_pred_instances):
            
            mask = pred_mask[:, :, i]
            #mask = mask[:, :, np.newaxis]
            #mask = np.repeat(mask, 3, axis=-1)
            cind, rind = np.where(mask == 1)
            coord_data = pred_coord[:, :, i, :].copy()
            coord_data[:, :, 2] = 1 - coord_data[:, :, 2] # undo the z axis flipping to match original data
            draw_image[cind, rind] = coord_data[cind, rind] * 255
            if draw_rgb_coord:
                b_image[cind, rind, 2] = coord_data[cind, rind, 2] * 255
                b_image[cind, rind, 0:2] = 0

                g_image[cind, rind, 1] = coord_data[cind, rind, 1] * 255
                g_image[cind, rind, 0] = 0
                g_image[cind, rind, 2] = 0

                r_image[cind, rind, 0] = coord_data[cind, rind, 0] * 255
                r_image[cind, rind, 1:3] = 0

        if draw_tag:
            for i in range(num_pred_instances):
                overlay = draw_image.copy()
                text = synset_names[pred_class_ids[i]]+'({:.2f})'.format(pred_scores[i])
                overlay = draw_text(overlay, pred_bbox[i], text, draw_box=True)
                cv2.addWeighted(overlay, alpha, draw_image, 1 - alpha, 0, draw_image)

        cv2.imwrite(output_path, draw_image[:, :, ::-1])

        plt.figure()
        plt.imshow(draw_image[:, :, ::-1])
        plt.savefig('hi.png')



        if draw_rgb_coord:
            cv2.imwrite(output_path_r, r_image[:, :, ::-1])
            cv2.imwrite(output_path_g, g_image[:, :, ::-1])
            cv2.imwrite(output_path_b, b_image[:, :, ::-1])
                        
        
        output_path = os.path.join(save_dir, '{}_{}_bbox_pred.png'.format(data_name, image_id))
        draw_image_bbox = image.copy()

        if gt_class_ids is not None:
            gt_match, pred_match, _, pred_indices = compute_matches(gt_bbox, gt_class_ids, gt_mask,
                                                                    pred_bbox, pred_class_ids, pred_scores, pred_mask,
                                                                    0.5)

            if len(pred_indices):
                pred_class_ids = pred_class_ids[pred_indices]
                pred_scores = pred_scores[pred_indices]        
                pred_RTs = pred_RTs[pred_indices]

        
        for ind in range(num_pred_instances):
            RT = pred_RTs[ind]
            cls_id = pred_class_ids[ind]
            
            if gt_class_ids is not None:## if gt exists, skip instances that fail to match
                gt_ind = int(pred_match[ind])
                if gt_ind == -1:
                    continue
            
            xyz_axis = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
            transformed_axes = transform_coordinates_3d(xyz_axis, RT)
            projected_axes = calculate_2d_projections(transformed_axes, intrinsics)


            bbox_3d = get_3d_bbox(pred_scales[ind, :], 0)
            transformed_bbox_3d = transform_coordinates_3d(bbox_3d, RT)
            projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
            draw_image_bbox = draw(draw_image_bbox, projected_bbox, projected_axes, (255, 0, 0))

        if draw_tag:
            if gt_class_ids is not None: ## if gt exists, draw rotation and translation error
                for ind in range(num_pred_instances):
                    gt_ind = int(pred_match[ind])
                    if gt_ind == -1:
                        continue

                    overlay = draw_image_bbox.copy()
                    RT = pred_RTs[ind]
                    gt_RT = gt_RTs[gt_ind]
                    cls_id = pred_class_ids[ind]
                    
                    degree, cm = compute_RT_degree_cm_symmetry(RT, gt_RT, cls_id, gt_handle_visibility, synset_names)
                    text = '{}({:.1f}, {:.1f})'.format(synset_names[cls_id], degree, cm)
                    overlay = draw_text(overlay, pred_bbox[ind], text)
                    cv2.addWeighted(overlay, alpha, draw_image_bbox, 1 - alpha, 0, draw_image_bbox)


        cv2.imshow("trst",draw_image_bbox[:, :, ::-1])
        cv2.waitKey(0)
        cv2.imwrite(output_path, draw_image_bbox[:, :, ::-1])

        plt.figure()
        plt.imshow(draw_image_bbox[:, :, ::-1])
        plt.savefig('hi.png')