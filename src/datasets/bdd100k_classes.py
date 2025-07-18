import json
from collections import Counter
import os
import numpy as np
import time
from logging import getLogger

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision




_GLOBAL_SEED = 0
logger = getLogger()

"""json_path = "/data/datasets/bdd100k/bdd100k/labels/bdd100k_labels_images_train.json"

with open(json_path, 'r') as f:
    data = json.load(f)
all_cats = []
print(f"Total number of entries in the dataset: {len(data)}")

for item in data:
    labels = item.get("labels", [])
    for label in labels:
        cat = label.get("category")
        if cat:
            all_cats.append(cat)
category_counts = Counter(all_cats)

print(f"Total unique categories: {len(category_counts)}")
for cat, count in category_counts.most_common():
    print(f"{cat}: {count}")"""
    
# Total number of entries in the dataset: 69863 , it should be 70000
# Total unique categories: 12
"""car: 713211
lane: 528643
traffic sign: 239686
traffic light: 186117
drivable area: 125723
person: 91349
truck: 29971
bus: 11672
bike: 7210
rider: 4517
motor: 3002
train: 136"""
# image root -> bdd100k/bdd100k/images/100k/train or bdd100k/bdd100k/images/100k/val


class BDDBoxDataset(Dataset):

    def __init__(
        self,
        root,
        transform=None,
        class_map=None,
        train=True,
        min_box_size=10,
    ):

        image_root = os.path.join(root, "images/100k/train") if train else os.path.join(root, "images/100k/val")
        annotation_file = os.path.join(root, "labels/bdd100k_labels_images_train.json") if train else os.path.join(root, "labels/bdd100k_labels_images_val.json") 
        self.transform = transform
        self.class_map = class_map or {
            "car": 0,
            "lane": 1,
            "traffic sign": 2,
            "traffic light": 3,
            "drivable area": 4,
            "person": 5,
            "truck": 6,
            "bus": 7,
            "bike": 8,
            "rider": 9,
            "motor": 10,
            "train": 11,
        }
        self.min_box_size = min_box_size
        self.samples = []

        with open(annotation_file, "r") as f:
            data = json.load(f)

        for entry in data:
            img_path = os.path.join(image_root, entry["name"])
            for obj in entry.get("labels", []):
                category = obj.get("category")
                if category not in self.class_map:
                    continue
                if "box2d" not in obj:
                    continue

                box = obj["box2d"]
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                if (x2 - x1) < min_box_size or (y2 - y1) < min_box_size:
                    continue

                self.samples.append({
                    "image_path": img_path,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "label": self.class_map[category],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        img = Image.open(entry["image_path"]).convert("RGB")
        crop = img.crop(entry["bbox"])
        if self.transform:
            crop = self.transform(crop)
        return crop, entry["label"]#, 0
    




def make_bddboxdataset(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    training=True,
    drop_last=True,
    persistent_workers=False,
    subset_file=None,
):
    dataset = BDDBoxDataset(
        root=root_path,
        transform=transform,
        train=training,
    )
    if subset_file is not None:
        raise NotImplementedError("Subset filtering not implemented for BDDBoxDataset")
    
    logger.info("BDDbox dataset created")
    dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset=dataset, num_replicas=world_size, rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    logger.info("BDDbox unsupervised data loader created")

    return dataset, data_loader, dist_sampler
