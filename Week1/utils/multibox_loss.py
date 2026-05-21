import torch.nn.functional as F
import torch.nn as nn

from prior_boxes    import prior_boxes, match, encode, decode

import numpy
import torch

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, overlap_threshold, neg_pos_ratio, variance):
        super(MultiBoxLoss, self).__init__()
        self.threshold     = overlap_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.variance      = variance
    
    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, priors = predictions
        gt_label_s, gt_box_s = targets
        
        device = loc_data.device
        
        batch_size  = loc_data .size( 0)
        num_priors  = loc_data .size( 1)
        num_classes = conf_data.size(-1)
            
        # match priors (default boxes) and ground truth boxes
        loc_t  = torch.zeros(batch_size, num_priors, 4, device=device).float()
        conf_t = torch.zeros(batch_size, num_priors   , device=device).long ()
            
        neg_pos_ratio = self.neg_pos_ratio
        threshold     = self.threshold
        variance      = self.variance
        
        for idx in range(batch_size):
            loc_t[idx], conf_t[idx] = match(threshold, gt_box_s[idx], priors, variance, gt_label_s[idx])
            
        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)
        
        # Localization Loss (Smooth L1)
        # Shape: [batch, num_priors, 4]
        loc_p = loc_data[pos].view(-1, 4)
        loc_t = loc_t   [pos].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')
        
        # Compute classification loss
        loss_c = F.cross_entropy(conf_data.view(-1, num_classes), conf_t.view(-1), reduction='none')
        loss_c = loss_c.view(batch_size, num_priors)
        
        # Filter out the negative samples and reduce the loss by sum
        loss_c_pos = loss_c[pos].sum()
        
        # Hard negative mining, filter out the positive samples and pick the
        # top negative losses
        num_neg = torch.clamp(neg_pos_ratio * num_pos, max=pos.size(1) - 1)
        loss_c_neg = loss_c * ~pos
        loss_c_neg, _ = loss_c_neg.sort(1, descending=True)
        neg_mask = torch.zeros_like(loss_c_neg)
        neg_mask[torch.arange(batch_size), num_neg.view(-1)] = 1.
        neg_mask = 1 - neg_mask.cumsum(-1)
        loss_c_neg = (loss_c_neg * neg_mask).sum()
        
        # Finally we normalize the losses by the number of positives
        N = num_pos.sum()
        loss_l = loss_l / N
        loss_c = (loss_c_pos + loss_c_neg) / N

        return loss_l, loss_c

if __name__ == '__main__':
    overlap_threshold = 0.5
    num_classes       = 3
    variance          = [0.1, 0.2] 
    neg_pos_ratio     = 3
    batch_size        = 1

    custom_config = {
     'num_classes': 3,
     'feature_maps' : [(45,80), (23,40), (12,20), (6,10), (3,5), (2,3)], #ResNet18
     'min_dim'      : 300,
     'min_sizes'    : [0.1, 0.20, 0.37, 0.54, 0.71, 1.00],
     'max_sizes'    : [0.2, 0.37, 0.54, 0.71, 1.00, 1.05],
     
     'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
     'variance'     : [0.1, 0.2],
     'clip'         : True,
    }
    
    prior_box_s = prior_boxes(custom_config)
    num_priors  = prior_box_s.shape[0]
    
    conf_s = torch.ones ( batch_size, num_priors, num_classes)
    loc_s  = torch.zeros( batch_size, num_priors,           4)
 
    gt_label_s = [ torch.from_numpy( numpy.array([1, 2]) ) ]
    gt_box_s   = [ torch.from_numpy( numpy.array([[0.0, 0.0, 0.5, 0.5],[0.5, 0.5, 1.0, 1.0]]) ) ]

    multi_box_loss = MultiBoxLoss(overlap_threshold, neg_pos_ratio, variance)
    
    print( multi_box_loss.forward((loc_s, conf_s, prior_box_s),(gt_label_s, gt_box_s)) )
