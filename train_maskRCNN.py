import torch
from torchvision import models, transforms
from torchvision.datasets import CocoDetection
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
import os

# Define the limited set of classes you want to train the model on
classes_to_keep = ['person', 'car', 'dog']

# Load the COCO dataset
coco_root = 'path/to/coco/root'
coco_annFile = os.path.join(coco_root, 'annotations', 'instances_train2017.json')
coco_dataset = CocoDetection(coco_root, coco_annFile)

# Filter the COCO dataset based on the required classes
coco = COCO(coco_annFile)
class_ids_to_keep = coco.getCatIds(catNms=classes_to_keep)
filtered_dataset = coco.loadImgs(coco.getImgIds(catIds=class_ids_to_keep))

# Create the model with the limited set of classes
model_mask_rcnn = models.detection.maskrcnn_resnet50_fpn(pretrained=False, num_classes=len(classes_to_keep)+1)

# Set up the training parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_mask_rcnn.to(device)
optimizer = torch.optim.SGD(model_mask_rcnn.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Train the model with the filtered dataset
num_epochs = 10
for epoch in range(num_epochs):
    for idx, (image, targets) in enumerate(filtered_dataset):
        image, targets = F.to_tensor(image).to(device), F.to_tensor(targets).to(device)
        model_mask_rcnn.train()
        optimizer.zero_grad()
        loss_dict = model_mask_rcnn(image, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

# Save the trained model
torch.save(model_mask_rcnn.state_dict(), 'mask_rcnn_limited_classes.pth')
