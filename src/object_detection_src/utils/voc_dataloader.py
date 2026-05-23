from torch.utils.data import DataLoader
from voc_dataset      import VOCDetection

import torch
import os

def default_collate(batch):
    label_ss, box_ss, image_s = [], [], []
    
    for sample in batch:
        image, box_s, label_s = sample
        
        if (len(box_s) > 0 and len(label_s) > 0) :
            image_s .append(image  )
            box_ss  .append(box_s  )
            label_ss.append(label_s)
    
    return torch.stack(image_s) if len(image_s) > 0 else torch.Tensor(), box_ss, label_ss


def get_train_dataloader(dataset_root_dir, annotation_file, batch_size, num_workers, sample_transform=None):
    from voc_dataset import strong_transform

    tr = strong_transform if sample_transform is None else sample_transform
    train_dataset = VOCDetection(dataset_root_dir, annotation_file, sample_transform=tr)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=default_collate,
        drop_last=True,
    )
    return train_dataloader


def get_test_dataloader(dataset_root_dir, annotation_file, batch_size, num_workers, sample_transform=None):
    from voc_dataset import strong_transform

    tr = strong_transform if sample_transform is None else sample_transform
    test_dataset = VOCDetection(dataset_root_dir, annotation_file, sample_transform=tr)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=default_collate,
    )
    return test_dataloader

if __name__ == '__main__':
    voc_root = "dataset"
    
    train_annotation_filename = os.path.join( voc_root, "ImageSets/Main/trainval.txt" )
    test_annotation_filename  = os.path.join( voc_root, "ImageSets/Main/test.txt"     )
    
    train_dataloader = get_train_dataloader(voc_root, train_annotation_filename, 1, 1) 
    
    for train_sample_s in train_dataloader:
        print(train_sample_s)
    
    test_dataloader  = get_test_dataloader(voc_root, test_annotation_filename, 1, 1)

    for test_sample_s in test_dataloader:
        print(test_sample_s)


