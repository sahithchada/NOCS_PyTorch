#used https://github.com/pytorch/vision/blob/main/torchvision/models/detection/mask_rcnn.py
import torch
import torchvision.models as models
import numpy as np
import os
import torch
import torchvision.transforms as T
from PIL import Image, ImageEnhance, ImageOps
import matplotlib.pyplot as plt
import random
# import sys
import argparse


# Define the function to apply the model to an input image and visualize the results
def visualize_segmentation(model, image_path, threshold=0.5):
    image = Image.open(image_path).convert('RGB')
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(image)
    
    with torch.no_grad():
        predictions = model([img_tensor])
    
    # Extract the predicted masks, labels, and scores
    masks = predictions[0]['masks']
    scores = predictions[0]['scores']

    for key,val in predictions[0].items():
        print(key,val.shape)

    # keys: ['boxes', 'labels', 'scores', 'masks']
    
    # Create a copy of the input image for visualization
    labels = predictions[0]['labels']
    vis_image = image.copy()    

    # Overlay the predicted masks with a transparency factor (alpha)
    full_img = torch.zeros_like(masks[0,0])
    for i in range(len(masks)):
        if scores[i] > threshold and labels[i] in valid_classes:
            mask = masks[i, 0].mul(255).byte().cpu().numpy()
            color = tuple(random.randint(0, 255) for _ in range(3))

            vis_image = apply_mask(vis_image, mask, color,alpha=0.6)

            full_img = full_mask(full_img,mask)

            print(predictions[0]['labels'][i])

    return vis_image, full_img

# Define the function to apply a mask to an image
def apply_mask(image, mask, color, alpha=0.5):
    mask_image = Image.fromarray(mask).convert('L')
    colored_mask = Image.new('RGBA', image.size, (0,0,0))
    
    # Create a solid mask for the colored_mask image
    solid_mask = Image.new('RGBA', image.size, color)
    colored_mask.paste(solid_mask, mask=mask_image)
    
    # Convert the original image to 'RGBA' mode and make sure both images have the same size
    image = image.convert('RGBA')
    colored_mask = colored_mask.resize(image.size)
    blended=Image.blend(image, colored_mask, alpha)
    
    im = Image.composite(blended, image, mask_image)
    
    return im

def full_mask(full_img,mask):

    full_img += mask

    return full_img

# Compute the Intersection over Union (IoU) between two masks
def iou(mask1, mask2):
    intersection = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    return intersection / union


def main():
    # Load the images from the test_data folder

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_dir", help="location of test data with respect to current folder",
                    type=str)
    args = parser.parse_args()

    color_image_paths = sorted([os.path.join(args.test_data_dir, img) for img in os.listdir(args.test_data_dir) if img.endswith('_color.png')])
    mask_image_paths = sorted([os.path.join(args.test_data_dir, img) for img in os.listdir(args.test_data_dir) if img.endswith('_mask.png')])

    model_mask_rcnn= models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    model_mask_rcnn.eval()

    # Perform the segmentation, visualize the results, and evaluate against ground truth masks
    iou_scores = []
    for color_image_path, mask_image_path in zip(color_image_paths, mask_image_paths):
        vis_image, mask = visualize_segmentation(model_mask_rcnn, color_image_path)
        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow(vis_image)
        plt.title(os.path.basename(color_image_path))
        plt.axis('off')
        
        # Load the ground truth mask and compute IoU
        gt_mask = Image.open(mask_image_path).convert('1')
        gt_mask_array = np.array(gt_mask)

        pred_mask_array = np.array(vis_image.convert('1'))

        mask = mask.cpu().detach().numpy()
        mask = mask > 0

        # print(mask[0])

        plt.subplot(2,2,2)
        plt.imshow(np.invert(mask))
        plt.subplot(2,2,3)
        plt.imshow(gt_mask_array)

        iou_score = iou(gt_mask_array, pred_mask_array)
        iou_scores.append(iou_score)
        print(f"IoU score for {os.path.basename(color_image_path)}: {iou_score:.4f}")

        # Display all the visualizations
        plt.show()

        # Compute and print the average IoU score
        average_iou = sum(iou_scores) / len(iou_scores)
        print(f"Average IoU score: {average_iou:.4f}")

if __name__ == "__main__":
    main()





#model_mask_rcnn= models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
#model_mask_rcnn.eval()
#in_features = model_mask_rcnn.roi_heads.box_predictor.cls_score.in_features
#x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)] #test
#predictions = model_mask_rcnn(x)
