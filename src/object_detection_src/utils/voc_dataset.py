import albumentations.pytorch.transforms

import xml.etree.ElementTree as ET
import torch.utils.data      as data
import albumentations        as A
import PIL.Image             as pilimg
import numpy                 as np

import torch
import cv2
import os

# Индекс 0 = background (так устроены match/detect_objects в prior_boxes.py).
# В XML размечены только car и license_plate → метки 1 и 2.
VOC_CLASSES = ('background', 'car', 'license_plate')

# VGG16 + датасет 640×360: задание — тренировать при 360×640.
IMAGE_RESOLUTION = (360, 640)
# ResNet18 SSD: задание — 1280×720 исходников, пайплайн к 720×1280 (H×W).
RESNET_IMAGE_RESOLUTION = (720, 1280)

# Синонимы имён из XML (нижний регистр → каноническое имя в VOC_CLASSES)
VOC_NAME_ALIASES = {
    'car': 'car',
    'license_plate': 'license_plate',
    'license plate': 'license_plate',
    'background': 'background',
}


bbox_params = A.BboxParams(format='albumentations', min_area=100, min_visibility=0.5, label_fields=['labels'])

light_transform = A.Compose([
    A.Resize(height=IMAGE_RESOLUTION[0], width=IMAGE_RESOLUTION[1], p=1.0),
    A.HorizontalFlip(p=0.5),
    A.RandomResizedCrop(size=IMAGE_RESOLUTION, p=0.5, scale=(0.7, 1.0)),
    A.GaussNoise(var_limit=(100, 150), p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True),
    A.pytorch.transforms.ToTensorV2()
], bbox_params=bbox_params,  p=1.0)

medium_transform = A.Compose([
    A.Resize(height=IMAGE_RESOLUTION[0], width=IMAGE_RESOLUTION[1], p=1.0),
    A.HorizontalFlip(p=0.5),
    A.RandomResizedCrop(size=IMAGE_RESOLUTION, p=0.5, scale=(0.7, 1.0)),
    A.MotionBlur(blur_limit=17, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True),
    A.pytorch.transforms.ToTensorV2()
], bbox_params=bbox_params, p=1.0)

strong_transform = A.Compose([
    A.Resize(height=IMAGE_RESOLUTION[0], width=IMAGE_RESOLUTION[1], p=1.0),
    A.RandomResizedCrop(IMAGE_RESOLUTION, p=0.5, scale=(0.7, 1.0)),
    A.RGBShift(p=0.5),
    A.Blur(blur_limit=11, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.CLAHE(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True),
    A.pytorch.transforms.ToTensorV2()
], bbox_params=bbox_params, p=1.0)

strong_transform_resnet = A.Compose([
    A.Resize(height=RESNET_IMAGE_RESOLUTION[0], width=RESNET_IMAGE_RESOLUTION[1], p=1.0),
    A.RandomResizedCrop(RESNET_IMAGE_RESOLUTION, p=0.5, scale=(0.7, 1.0)),
    A.RGBShift(p=0.5),
    A.Blur(blur_limit=11, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.CLAHE(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True),
    A.pytorch.transforms.ToTensorV2()
], bbox_params=bbox_params, p=1.0)

class VOCDetection(data.Dataset):
    def __init__(self, voc_root, annotation_filename, sample_transform=strong_transform):
        self.annotation_filename = annotation_filename
        self.sample_transform    = sample_transform
        self.root                = voc_root  
        
        self.id_s = []
        with open(self.annotation_filename, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                ano_path = os.path.join(self.root, 'Annotations', line + '.xml')
                img_path = os.path.join(self.root, 'JPEGImages' , line + '.jpg')
                
                boxes, labels, (height, width) = self.__load_annotation(ano_path)
                if len(boxes) > 0 and len(labels) > 0:
                    self.id_s.append((img_path, ano_path))
    
    def __load_annotation(self, path):
        tree = ET.parse(path)
        
        size_node = tree.find('size')
        height = float(size_node.find('height').text)
        width  = float(size_node.find('width' ).text)
        
        boxes, labels  = [], []
        for child in tree.getroot():
            if child.tag == 'object':
                bndbox = child.find('bndbox')
                box = [float(bndbox.find(t).text) - 1 for t in ['xmin', 'ymin', 'xmax', 'ymax']]
                raw_name = child.find('name').text.strip()
                canon = VOC_NAME_ALIASES.get(raw_name.lower(), raw_name)
                if canon in VOC_CLASSES:
                    label = VOC_CLASSES.index(canon)
                    labels.append(label)
                    boxes.append(box)
        
        return np.array(boxes), np.array(labels), (height, width)
    
    def __load_image(self, path):
        image = cv2.imread(path   , cv2.IMREAD_COLOR )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def __getitem__(self, index):
        img_path, ano_path = self.id_s[index]
        
        source_box_s, source_label_s, (height, width) = self.__load_annotation(ano_path)
        source_image                                  = self.__load_image     (img_path)
        
        source_box_s = np.clip( source_box_s / np.array([width-1, height-1, width-1, height-1]), 0, 1)
        #source_box_s = np.clip( source_box_s / np.array([1280-1, 720-1, 1280-1, 720-1]), 0, 1)
        
        result = self.sample_transform(image=source_image, bboxes=source_box_s, labels=source_label_s )
        target_image   =          result['image' ]
        target_box_s   = np.array(result['bboxes'])
        target_label_s = np.array(result['labels'])
        
        #target_image   = torch.from_numpy( np.transpose( (source_image.astype(np.float32)/255), axes=(2,0,1) ))
        #target_box_s   = source_box_s
        #target_label_s = source_label_s
        
        if isinstance(target_image, torch.Tensor):
            target_image = target_image.detach().clone()
        else:
            target_image = torch.as_tensor(target_image)
        return target_image, torch.from_numpy(target_box_s), torch.from_numpy(target_label_s)
    
    def __len__(self):
        return len(self.id_s)
 
if __name__ == '__main__':
    voc_root = "dataset"
    
    train_annotation_filename = os.path.join( voc_root, "ImageSets/Main/trainval.txt" )
    test_annotation_filename  = os.path.join( voc_root, "ImageSets/Main/test.txt"     )
    
    train_dataset = VOCDetection( voc_root, train_annotation_filename )
    
    for train_sample in train_dataset:
        print( train_sample )
    
    test_dataset  = VOCDetection( voc_root, test_annotation_filename  )

    for test_sample in test_dataset:
        print(test_sample)
