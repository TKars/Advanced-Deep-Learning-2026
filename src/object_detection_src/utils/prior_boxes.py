from itertools import product
from math      import sqrt

import torchvision
import numpy
import torch

def prior_boxes(cfg):
    num_priors    = len(cfg['aspect_ratios'])
    variance      = cfg['variance'] or [0.1, 0.2]
    feature_maps  = cfg['feature_maps' ]
    min_sizes     = cfg['min_sizes'    ]
    max_sizes     = cfg['max_sizes'    ]
    aspect_ratios = cfg['aspect_ratios']
    clip          = cfg['clip'         ]
        
    prior_box_s = []
    for k, feature_map in enumerate(feature_maps):
        for i, j in product(range(feature_map[0]), range(feature_map[1])):
            # unit center x,y
            cx = (j + 0.5) / feature_map[1]
            cy = (i + 0.5) / feature_map[0]
            # aspect_ratio: 1
            # rel size: min_size
            s_k = min_sizes[k]
            prior_box_s += [cx, cy, s_k, s_k]
            # aspect_ratio: 1
            # rel size: sqrt(s_k * s_(k+1))
            s_k_prime = sqrt(min_sizes[k] * (max_sizes[k]))
            prior_box_s += [cx, cy, s_k_prime, s_k_prime]
            # rest of aspect ratios
            for ar in aspect_ratios[k]:
                prior_box_s += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                prior_box_s += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
    # back to torch land
    prior_box_s = torch.Tensor(prior_box_s).view(-1, 4)
    if clip:
        prior_box_s.clamp_(max=1, min=0)
    return prior_box_s

def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2, boxes[:, :2] + boxes[:, 2:]/2), 1)  #xmin, ymin xmax, ymax

def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2, boxes[:, 2:] - boxes[:, :2], 1)  # cx, cy, w, h

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A, B = box_a.size(0), box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def match(threshold, truths, priors, variances, labels):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    overlaps = jaccard(truths, point_form(priors))
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_    (0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_    (1)
    best_prior_overlap.squeeze_(1)

    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf    = labels[best_truth_idx]          # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background
    loc = encode(matches, priors, variances)
    return loc, conf   # [num_priors,4] encoded offsets to learn # [num_priors] top class label for each prior

def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """
    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]

def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = torch.cat(( priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:], priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def detect_objects(loc_data, conf_data, prior_data, num_classes, overlap_threshold, conf_threshold ):
    """
    Args:
        loc_data: (tensor) Loc preds from loc layers
            Shape: [batch,num_priors*4]
        conf_data: (tensor) Shape: Conf preds from conf layers
            Shape: [batch,num_priors,num_classes]
        prior_data: (tensor) Prior boxes and variances from priorbox layers
            Shape: [num_priors,4]
    Return:
        box_data: (list)
            Shape: batch, [num_objects, 4]
        label_data: (list)
            Shape: batch, [num_objects]
        conf_data: (list)
            Shape: batch, [num_objects]
    """
    variance = [0.1, 0.2]
    device = loc_data.device
    
    num = loc_data.size(0) # batch size
    num_priors = prior_data.size(0)
    
    result_box_ss, result_conf_ss, result_label_ss = [], [], []
    conf_preds = conf_data.transpose(2, 1)
    # Decode predictions into bboxes.
    for i in range(num):
        decoded_boxes = decode(loc_data[i], prior_data, torch.as_tensor(variance))
        # For each class, perform nms
        conf_scores = conf_preds[i].clone()
        
        result_box_s, result_conf_s, result_label_s = [], [], []
        
        for cl in range(1, num_classes):
            c_mask = conf_scores[cl].gt(conf_threshold)
            scores = conf_scores[cl][c_mask]
            if scores.size(0) == 0:
                continue
            
            l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
            boxes = decoded_boxes[l_mask].view(-1, 4)
            # idx of highest scoring and non-overlapping boxes per class
            ids = torchvision.ops.nms(boxes, scores, overlap_threshold)
            count = ids.size(0)
            
            result_box_s  .append(boxes [ids[:count]])
            result_conf_s .append(scores[ids[:count]])
            result_label_s.append(torch.full((count,), cl))
        
        result_box_ss  .append(torch.cat(result_box_s  , dim=0) if len(result_box_s  ) > 0 else torch.Tensor(0) )
        result_conf_ss .append(torch.cat(result_conf_s , dim=0) if len(result_conf_s ) > 0 else torch.Tensor(0) )
        result_label_ss.append(torch.cat(result_label_s, dim=0) if len(result_label_s) > 0 else torch.Tensor(0) )
    
    return result_box_ss, result_label_ss, result_conf_ss

if __name__ == '__main__':
    custom_config = {
     'num_classes': 3,
     'feature_maps' : [(45,80), (23,40), (12,20), (6,10), (3,5), (2,3)], #ResNet18
     'min_dim'      : 300,
     'min_sizes'    : [0.1, 0.20, 0.37, 0.54, 0.71, 1.00],
     'max_sizes'    : [0.2, 0.37, 0.54, 0.71, 1.00, 1.05],
     
     'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
     'variance'     : [0.1, 0.2],
     'clip'         :    True,
    }
    
    threshold = 0.9
    variance  = [0.1, 0.2] 

    prior_box_s = prior_boxes(custom_config)
    prior_loc_s = torch.zeros(prior_box_s.shape)
    
    result_box_s = decode(prior_loc_s, prior_box_s , variance)
    
    result_loc_s = encode(result_box_s, prior_box_s, variance)
    
    gt_label_s = torch.from_numpy( numpy.array([1, 2]) ) 
    gt_box_s   = torch.from_numpy( numpy.array([[0.0, 0.0, 0.5, 0.5],[0.5, 0.5, 1.0, 1.0]]) ) 
    
    num_gt_objects = 2
    num_priors     = prior_box_s.shape[0]
    
    loc, conf = match(threshold, gt_box_s, prior_box_s, variance, gt_label_s)
