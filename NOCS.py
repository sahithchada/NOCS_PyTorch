import torch
import torch.nn as nn
from utils import SamePad2d, pyramid_roi_align

class Nocs_head_bins_wt_unshared(nn.Module):
    def __init__(self,depth, pool_size,image_shape, num_classes, num_bins, net_name):
        super(Nocs_head_bins_wt_unshared, self).__init__()
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.num_bins=num_bins
        self.net_name=net_name
        self.depth=depth

        self.padding = SamePad2d(kernel_size=3, stride=1)
        self.conv1 = nn.Conv2d(self.depth, 256, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(256, eps=0.001)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(256, eps=0.001)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(256, eps=0.001)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(256, eps=0.001)
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256, self.num_bins * self.num_classes, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, rois):
        x = pyramid_roi_align([rois] + x, self.pool_size, self.image_shape)
        x = self.conv1(self.padding(x))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(self.padding(x))
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(self.padding(x))
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(self.padding(x))
        x = self.bn4(x)
        x = self.relu(x)
        x_feature = self.deconv(x)
        x = self.relu(x_feature)
        x = self.conv5(x)

        x=x.view(x.shape[0], -1,self.num_bins, x.shape[2], x.shape[3])
        x = self.softmax(x)

        return x,x_feature
    
class CoordBinValues(nn.Module):
    def __init__(self, coord_num_bins):
        super(CoordBinValues, self).__init__()
        self.coord_num_bins = coord_num_bins

    def forward(self, mrcnn_coord_bin):
        mrcnn_coord_shape = mrcnn_coord_bin.shape
        #mrcnn_coord_bin_reshape = mrcnn_coord_bin.view(-1, mrcnn_coord_shape[-1])
        mrcnn_coord_bin_reshape = mrcnn_coord_bin.view(-1, mrcnn_coord_shape[2])

        mrcnn_coord_bin_ind = torch.argmax(mrcnn_coord_bin_reshape, dim=-1)
        mrcnn_coord_bin_value = mrcnn_coord_bin_ind.float() / self.coord_num_bins
        #mrcnn_coord_bin_value = mrcnn_coord_bin_value.view(*mrcnn_coord_shape[0,2,3,4])
        mrcnn_coord_bin_value = mrcnn_coord_bin_value.view(mrcnn_coord_shape[0], mrcnn_coord_shape[1], mrcnn_coord_shape[3], mrcnn_coord_shape[4])

        return mrcnn_coord_bin_value


