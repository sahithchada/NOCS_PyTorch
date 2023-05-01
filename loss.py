import torch
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

############################################################
#  Loss Functions
############################################################

def compute_losses(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask,pred_coords,target_coords,target_domain_labels):

    rpn_class_loss = compute_rpn_class_loss(rpn_match, rpn_class_logits)
    rpn_bbox_loss = compute_rpn_bbox_loss(rpn_bbox, rpn_match, rpn_pred_bbox)
    mrcnn_class_loss = compute_mrcnn_class_loss(target_class_ids, mrcnn_class_logits)
    mrcnn_bbox_loss = compute_mrcnn_bbox_loss(target_deltas, target_class_ids, mrcnn_bbox)
    mrcnn_mask_loss = compute_mrcnn_mask_loss(target_mask, target_class_ids, mrcnn_mask)
    mrcnn_coord_bins_symmetry_loss=compute_mrcnn_coord_bins_symmetry_loss(target_mask, target_coords, target_class_ids, target_domain_labels, pred_coords)

    # target_mask_np = target_mask.detach().cpu().numpy()
    # target_coords_np = target_coords.detach().cpu().numpy()
    # target_class_ids_np = target_class_ids.detach().cpu().numpy()
    # target_domain_labels_np = target_domain_labels.detach().cpu().numpy()
    # pred_coords_np = pred_coords.detach().cpu().numpy()
    # np.savez('loss_inputs.npz', target_mask_np = target_mask_np , target_coords_np=target_coords_np, target_class_ids_np = target_class_ids_np, target_domain_labels_np = target_domain_labels_np, pred_coords_np = pred_coords_np)


    return [rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss,mrcnn_coord_bins_symmetry_loss]

def nocs_map_rotation(positive_class_ids,positive_ix,target_coords,mask_shape):
    positive_class_rotation_theta = torch.tensor([class_id_to_theta(x) for x in positive_class_ids], dtype=torch.float32)

    positive_class_rotation_matrix = torch.stack([rotation_y_matrix(x) for x in positive_class_rotation_theta]).reshape(-1, 3, 3)
    positive_class_rotation_matrix = positive_class_rotation_matrix.reshape(-1, 1, 1, 3, 3) # [num_pos_rois, 1, 1, 3, 3]

    tiled_rotation_matrix = positive_class_rotation_matrix.repeat(1, mask_shape[2], mask_shape[3], 1, 1)
    indices = torch.stack([positive_ix, positive_class_ids], dim=1)

    if indices.is_cuda:
        tiled_rotation_matrix = tiled_rotation_matrix.cuda()

    y_true = target_coords[positive_ix] - 0.5


    y_true = y_true.unsqueeze(4)

    ## num_rotations = 6
    rotated_y_true_1 = torch.matmul(tiled_rotation_matrix, y_true)
    rotated_y_true_2 = torch.matmul(tiled_rotation_matrix, rotated_y_true_1)
    rotated_y_true_3 = torch.matmul(tiled_rotation_matrix, rotated_y_true_2)
    rotated_y_true_4 = torch.matmul(tiled_rotation_matrix, rotated_y_true_3)
    rotated_y_true_5 = torch.matmul(tiled_rotation_matrix, rotated_y_true_4)

    # Gather the coordinate maps and masks (predicted and true) that contribute to loss
    # true coord map:[N', height, width, bins]
    y_true_stack = torch.cat([y_true, rotated_y_true_1, rotated_y_true_2, rotated_y_true_3, rotated_y_true_4, rotated_y_true_5], dim=4)


    return y_true_stack, indices

def compute_mrcnn_coord_bins_symmetry_loss(target_masks, target_coords, target_class_ids, target_domain_labels, pred_coords):
    """Mask L2 loss for the coordinates head.
    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_coords: [batch, num_rois, height, width, 3].
        A float32 tensor of values in the range of [0, 1]. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_coords: [batch, proposals, height, width, num_classes, num_bins, 3] float32 tensor with values from 0 to 1.
    """

    if target_class_ids.size()[0]:
        # Only positive ROIs contribute to the loss. And only
        # the class specific mask of each ROI.

        #transforms to match the required input dimentions
        target_coords=torch.permute(target_coords,(1,2,3,0))
        pred_coords=torch.permute(pred_coords,(1,4,5,2,3,0))

        # mask_numpy= target_masks.detach().cpu().numpy()
        # coord_numpy=target_coords.detach().cpu().numpy()
        # for i in range(target_masks.shape[0]):
        #     print(target_class_ids[i])
        #     cv2.imshow("coords_x",coord_numpy[i,:,:,0])
        #     cv2.imshow("coords_y",coord_numpy[i,:,:,1])
        #     cv2.imshow("coords_z",coord_numpy[i,:,:,2])
        #     cv2.imshow("mask",mask_numpy[i])
        #     cv2.imshow("coords_all",coord_numpy[i])
        #     cv2.waitKey(0)

        target_masks=torch.unsqueeze(target_masks, 0)

        # Reshape for simplicity. Merge first two dimensions into one.

        num_bins = pred_coords.size(-2) 

        target_class_ids = target_class_ids.view(-1,)
        mask_shape = target_masks.size()
        target_masks = target_masks.view(-1, mask_shape[2], mask_shape[3])
        target_coords = target_coords.view(-1, mask_shape[2], mask_shape[3], 3)

        pred_shape = pred_coords.size()
        pred_coords_reshape = pred_coords.view(-1, pred_shape[1], pred_shape[2], pred_shape[3], num_bins, 3)

        # Permute predicted coords to [N, num_classes, height, width, 3, num_bins]
        pred_coords_trans = pred_coords_reshape.permute(0, 3, 1, 2, 5, 4)

        # Only positive ROIs contribute to the loss. And only the class specific mask of each ROI.
        # Only ROIs from synthetic images have the ground truth coord map and therefore contribute to the loss.
        target_domain_labels = target_domain_labels.view(-1,)
        domain_ix = torch.eq(target_domain_labels, False)
        target_class_ids = torch.mul(target_class_ids, domain_ix.float())

        positive_ix = torch.nonzero(target_class_ids > 0, as_tuple=True)[0]


        def nonzero_positive_loss(target_masks, target_coords, pred_coords_trans, positive_ix):
            #positive_class_ids = torch.tensor(target_class_ids)[positive_ix].to(torch.int64)
            positive_class_ids = target_class_ids[positive_ix].to(torch.int64)

            y_true_stack , indices = nocs_map_rotation(positive_class_ids,positive_ix,target_coords,mask_shape)
            ## shape: [num_pos_rois, height, width, 3, 6] 
            
            
            y_true_stack = y_true_stack.permute(0, 1, 2, 4, 3)## shape: [num_pos_rois, height, width, 6, 3]
            y_true_stack = y_true_stack + 0.5

            target_mask_positive_index=target_masks[:y_true_stack.shape[0],:]
            expanded_mask=target_mask_positive_index.unsqueeze(-1).unsqueeze(-1).expand_as(y_true_stack)
            masked_y_true_stack=expanded_mask*y_true_stack
            # y_true_stack_numpy=y_true_stack.detach().cpu().numpy()
            # mask_numpy= target_masks.detach().cpu().numpy()
            # for i in range(y_true_stack.shape[0]):
            #     print(target_class_ids[i])
            #     cv2.imshow("mask",mask_numpy[i])
            #     for j in range(y_true_stack_numpy.shape[3]):
            #         squeezed=y_true_stack_numpy[i,:,:,j,:]

            #         cv2.imshow("coords"+str(j),squeezed)

            #     cv2.waitKey(0)

            y_true_bins_stack = y_true_stack * float(num_bins) - 1e-6
            y_true_bins_stack = torch.floor(y_true_bins_stack)
            y_true_bins_stack = y_true_bins_stack.to(torch.int64)

            y_true_bins_stack = torch.clamp(y_true_bins_stack, min=0, max=num_bins-1)


            y_pred = gather_nd_torch(pred_coords_trans, indices)
            y_pred = y_pred.unsqueeze(3)  # shape: [num_pos_roi, height, width, 1, 3, num_bins]

            # Tile y_pred to match the shape of y_true_stack
            y_pred_stack = y_pred.repeat(1, 1, 1, y_true_stack.size(3), 1, 1)

            y_pred_logits=torch.log(y_pred_stack+1e-5).permute(0,5,1,2,3,4)
            cross_loss = F.nll_loss(y_pred_logits, y_true_bins_stack,reduction='none')

            mask = torch.index_select(target_masks, 0, positive_ix) ## shape: [num_pixels_in_mask, 6, 3]
            mask = torch.index_select(target_masks, 0, positive_ix) ## shape: [num_pixels_in_mask, 6, 3]
            reshape_mask = mask.reshape(mask.shape[0], mask.shape[1], mask.shape[2], 1, 1) 
            ## shape: [num_pos_rois, height, width, 1, 1]

            num_of_pixels = mask.sum(dim=[1, 2]) + 0.00001 ## shape: [num_pos_rois]

            cross_loss_in_mask = cross_loss * reshape_mask
            #cross_loss_in_mask = cross_loss 
            sum_loss_in_mask = cross_loss_in_mask.sum(dim=[1,2])
            total_sum_loss_in_mask = sum_loss_in_mask.sum(dim=-1)

            arg_min_rotation = torch.argmin(total_sum_loss_in_mask,dim=-1).to(torch.int32)

            min_indices = torch.stack([torch.arange(arg_min_rotation.shape[0],device = arg_min_rotation.device), arg_min_rotation], dim=-1)
            min_loss_in_mask = gather_nd_torch(sum_loss_in_mask, min_indices)

            mean_loss_in_mask = min_loss_in_mask /  num_of_pixels.unsqueeze(1)

            sym_loss = mean_loss_in_mask.mean(0)

            return sym_loss

        if positive_ix.numel() > 0:
            loss = nonzero_positive_loss(target_masks, target_coords, pred_coords_trans, positive_ix)
        else:
            loss = torch.tensor([0.0, 0.0, 0.0])
        
    
    else:
        loss = torch.FloatTensor([[0],[0],[0]])
        if target_class_ids.is_cuda:
            loss = loss.cuda()
    
    return loss   



def compute_rpn_class_loss(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """

    # Squeeze last dim to simplify
    rpn_match = rpn_match.squeeze(2)

    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = (rpn_match == 1).long()

    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = torch.nonzero(rpn_match != 0)

    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = rpn_class_logits[indices.data[:,0],indices.data[:,1],:]
    anchor_class = anchor_class[indices.data[:,0],indices.data[:,1]]

    # Crossentropy loss
    loss = F.cross_entropy(rpn_class_logits, anchor_class)

    return loss

def compute_rpn_bbox_loss(target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.

    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """

    # Squeeze last dim to simplify
    rpn_match = rpn_match.squeeze(2)

    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    indices = torch.nonzero(rpn_match==1)

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = rpn_bbox[indices.data[:,0],indices.data[:,1]]

    # Trim target bounding box deltas to the same length as rpn_bbox.
    target_bbox = target_bbox[0,:rpn_bbox.size()[0],:]

    # Smooth L1 loss
    loss = F.smooth_l1_loss(rpn_bbox, target_bbox)

    return loss


def compute_mrcnn_class_loss(target_class_ids, pred_class_logits):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    """

    # Loss
    if target_class_ids.size()[0]:
        loss = F.cross_entropy(pred_class_logits,target_class_ids.long())
    else:
        loss = torch.FloatTensor([0])
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss


def compute_mrcnn_bbox_loss(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """

    if target_class_ids.size()[0]:
        # Only positive ROIs contribute to the loss. And only
        # the right class_id of each ROI. Get their indicies.
        positive_roi_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_roi_class_ids = target_class_ids[positive_roi_ix.data].long()
        indices = torch.stack((positive_roi_ix,positive_roi_class_ids), dim=1)

        # Gather the deltas (predicted and true) that contribute to loss
        target_bbox = target_bbox[indices[:,0].data,:]
        pred_bbox = pred_bbox[indices[:,0].data,indices[:,1].data,:]

        # Smooth L1 loss
        loss = F.smooth_l1_loss(pred_bbox, target_bbox)
    else:
        loss = torch.FloatTensor([0])
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss


def compute_mrcnn_mask_loss(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """



    if target_class_ids.size()[0]:
        # Only positive ROIs contribute to the loss. And only
        # the class specific mask of each ROI.
        positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_class_ids = target_class_ids[positive_ix.data].long()
        indices = torch.stack((positive_ix, positive_class_ids), dim=1)

        # Gather the masks (predicted and true) that contribute to loss
        y_true = target_masks[indices[:,0].data,:,:]
        y_pred = pred_masks[indices[:,0].data,indices[:,1].data,:,:]

        # Binary cross entropy
        loss = F.binary_cross_entropy(y_pred, y_true)
    else:
        loss = torch.FloatTensor([0])
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss

def class_id_to_theta(class_id):
    def my_func(class_id):
        target_classes = torch.tensor([1, 2, 4], device=class_id.device)
        is_in_target = (class_id[..., None] == target_classes).any(-1)
        result = torch.where(is_in_target, torch.tensor(2 * math.pi / 6, dtype=torch.float32, device=class_id.device), torch.tensor(0, dtype=torch.float32, device=class_id.device))
        return result

    return my_func(class_id)


def rotation_y_matrix(theta):
    rotation_matrix = torch.stack([torch.cos(theta), torch.tensor(0.0), torch.sin(theta),
                                    torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0),
                                    -torch.sin(theta), torch.tensor(0.0), torch.cos(theta)])
    rotation_matrix = rotation_matrix.reshape(3, 3)
    return rotation_matrix

def gather_nd_torch(params, indices, batch_dims=0):
    """ The same as tf.gather_nd.
    indices is an k-dimensional integer tensor, best thought of as a (k-1)-dimensional tensor of indices into params, where each element defines a slice of params:

    output[\\(i_0, ..., i_{k-2}\\)] = params[indices[\\(i_0, ..., i_{k-2}\\)]]

    Args:
        params (Tensor): "n" dimensions. shape: [x_0, x_1, x_2, ..., x_{n-1}]
        indices (Tensor): "k" dimensions. shape: [y_0,y_2,...,y_{k-2}, m]. m <= n.

    Returns: gathered Tensor.
        shape [y_0,y_2,...y_{k-2}] + params.shape[m:] 

    """
    if isinstance(indices, torch.Tensor):
      indices = indices.cpu().numpy()
    else:
      if not isinstance(indices, np.array):
        raise ValueError(f'indices must be `torch.Tensor` or `numpy.array`. Got {type(indices)}')
    if batch_dims == 0:
        orig_shape = list(indices.shape)
        num_samples = int(np.prod(orig_shape[:-1]))
        m = orig_shape[-1]
        n = len(params.shape)

        if m <= n:
            out_shape = orig_shape[:-1] + list(params.shape[m:])
        else:
            raise ValueError(
                f'the last dimension of indices must less or equal to the rank of params. Got indices:{indices.shape}, params:{params.shape}. {m} > {n}'
            )
        indices = indices.reshape((num_samples, m)).transpose().tolist()
        output = params[indices]    # (num_samples, ...)
        return output.reshape(out_shape).contiguous()
    else:
        batch_shape = params.shape[:batch_dims]
        orig_indices_shape = list(indices.shape)
        orig_params_shape = list(params.shape)
        assert (
            batch_shape == indices.shape[:batch_dims]
        ), f'if batch_dims is not 0, then both "params" and "indices" have batch_dims leading batch dimensions that exactly match.'
        mbs = np.prod(batch_shape)
        if batch_dims != 1:
            params = params.reshape(mbs, *(params.shape[batch_dims:]))
            indices = indices.reshape(mbs, *(indices.shape[batch_dims:]))
        output = []
        for i in range(mbs):
            output.append(gather_nd_torch(params[i], indices[i], batch_dims=0))
        output = torch.stack(output, dim=0)
        output_shape = orig_indices_shape[:-1] + list(orig_params_shape[orig_indices_shape[-1]+batch_dims:])
        return output.reshape(*output_shape).contiguous()