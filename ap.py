#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import random
import math
import re
import time
import numpy as np
import cv2
import skimage
import json
import matplotlib
import matplotlib.pyplot as plt

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Directory to load source dataset
DATA_DIR = os.path.join(ROOT_DIR, "../data/leftImg8bit")

# Directory to load groundtruth dataset
MASK_DIR = os.path.join(ROOT_DIR, "../data/gtFine")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

TRAINING = True
subset = ''


class CityscapeConfig(Config):
    """Configuration for training on the cityscape dataset.
    Derives from the base Config class and overrides values specific
    to the cityscape dataset.
    """
    # Give the configuration a recognizable name
    NAME = "cityscape"

    # We use a GPU with 12GB memory.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1  # 8

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000

    # Number of validation steRPNps to run at the end of every training epoch.
    VALIDATION_STEPS = 50

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 15  # background + 1 shapes
    GRADIENT_CLIP_NORM = 0.001

    # Input image resing
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024

    # Learning rate and momentum
    LEARNING_RATE = 0.0005


config = CityscapeConfig()
# config.display()


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

import logging
logger = logging.getLogger(__name__)
import os
class _PathManager():
	def __init__(self):
		pass
	@classmethod
	def ls(cls, DIR):
		return os.listdir(DIR)
	@classmethod
	def cat(cls, DIR1, DIR2):
		return os.path.join(DIR1, DIR2)
PathManager = _PathManager()

class CityscapeDataset(utils.Dataset):
    '''
    load_shapes()
    load_image()
    load_mask()
    '''

    def __init__(self, subset):
        super(CityscapeDataset, self).__init__(self)
        self.subset = subset

    def load_shapes(self):
        """
        subset: "train"/"val"
        image_id: use index to distinguish the images.
        gt_id: ground truth(mask) id.
        height, width: the size of the images.
        path: directory to load the images.
        """
        # Add classes you want to train
        self.add_class("cityscape",  1, "bicycle")
        self.add_class("cityscape",  2, "motorcycle")
        self.add_class("cityscape",  3, "train")
        self.add_class("cityscape",  4, "bus")
        self.add_class("cityscape",  5, "truck")
        self.add_class("cityscape",  6, "car")
        self.add_class("cityscape",  7, "person")
        self.add_class("cityscape",  8, "sky")
        self.add_class("cityscape",  9, "vegetation")
        self.add_class("cityscape", 10, "traffic sign")
        self.add_class("cityscape", 11, "traffic light")
        self.add_class("cityscape", 12, "building")
        self.add_class("cityscape", 13, "sidewalk")
        self.add_class("cityscape", 14, "road")
        self.add_class("cityscape", 15, "ground")

        # Add images
        image_dir = PathManager.cat(DATA_DIR, self.subset)
        cities = PathManager.ls(image_dir)
        
        image_names = []
        image_paths = []
        for city in cities:
            city_img_dir = PathManager.cat(image_dir, city)
            for basename in PathManager.ls(city_img_dir):
                image_file = PathManager.cat(city_img_dir, basename)
                
                suffix = "_leftImg8bit.png"
                assert basename.endswith(suffix), basename
                basename = os.path.basename(basename)[: -len(suffix)]
                
                image_paths.append(image_file)
                image_names.append(basename)

        image_dir = "{}/{}".format(DATA_DIR, self.subset)
        image_ids = os.listdir(image_dir)

        for idx in range(len(image_names)):
            #if(idx + 1) % 100 == 0:
            #    print("loading shape of {}th image, subset = {}".format(idx + 1, self.subset))
            self.add_image(source="cityscape", image_id = idx, gt_id = image_names[idx],
                height=1024, width=2048,
                path = image_paths[idx])

        #for index, item in enumerate(image_ids):
        #    if (index+1) %100 == 0:
        #        print("loading shape of {}th image, subset = {}".format(index + 1, self.subset))
        #    temp_image_path = "{}/{}".format(image_dir, item)
        #    temp_image_size = skimage.io.imread(temp_image_path).shape
        #    self.add_image("cityscape", image_id=index, gt_id=os.path.splitext(item)[0],
        #                    height=temp_image_size[0], width=temp_image_size[1],
        #                    path=temp_image_path)
        
    def load_image(self, image_id):
        """Load images according to the given image ID."""
        info = self.image_info[image_id]
        image = skimage.io.imread(info['path'])
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cityscape":
            return info["cityscape"]
        else:
            super(self.__class__).image_reference(self, image_id)

    #def load_mask(self, image_id):
    #    pass
    def load_mask(self, image_id):
        """Load instance masks of the given image ID.
        count: the number of masks in each image.
        class_id: the first letter of each mask file's name.
        """
        info = self.image_info[image_id]
        mask_dir = info['path'].replace("leftImg8bit", "gtFine", 1)[:-len("_leftImg8bit.png")]+"_instance"
        masks_list = os.listdir(mask_dir)
        count = len(masks_list)
        mask = np.zeros([info['height'], info['width'], count])
        class_ids = []

        for index, item in enumerate(masks_list):
            temp_mask_path = "{}/{}".format(mask_dir, item)
            tmp_mask = 255 - skimage.io.imread(temp_mask_path)[:, :, np.newaxis]
            mask[:, :, index:index+1] = tmp_mask
            class_ids.append(item.split("_")[0])

        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        return mask, np.array(class_ids, dtype=np.uint8)

    def load_semantic(self, image_id):
        info = self.image_info[image_id]
        semantic_dir = info['path'].replace("leftImg8bit", "gtFine", 1)[:-len("_leftImg8bit.png")]+"_semantic"
        semantic_list = os.listdir(semantic_dir)

        semantic = np.zeros([info['height'], info['width'], config.NUM_CLASSES])
        for sem in semantic_list:
            category_id = int(sem[:-len(".png")])
            tmp_sem_path = os.path.join(semantic_dir, sem)
            tmp_semantic = 255 - skimage.io.imread(tmp_sem_path)
            semantic[:, :, category_id] = tmp_semantic
        return semantic

    def load_depth(self, image_id):
        info = self.image_info[image_id]

        depth_dir = info['path'].replace("leftImg8bit", "disparity", 1).replace("leftImg8bit", "disparity")
        depth_map = skimage.io.imread(depth_dir)

        camera_dir = info['path'].replace("leftImg8bit", "camera", 1).replace("leftImg8bit", "camera").replace(".png", ".json")
        with open(camera_dir) as f:
            camera = json.load(f)
        baseline        = camera['extrinsic']['baseline']
        focal_length    = camera['intrinsic']['fx']

        depth_map = (depth_map.astype(np.float32) - 1.) / 256.
        NaN = depth_map <= 0

        depth_map[NaN] = 1
        depth_map       = baseline * focal_length / depth_map
        depth_map[NaN] = 0

        return depth_map

# Validation dataset
dataset_val = CityscapeDataset("val")
dataset_val.load_shapes()
dataset_val.prepare()

class InferenceConfig(CityscapeConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
print(model.find_last())
model_path = model.find_last()

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)

model.load_weights(model_path, by_name=True)

# Compute mAP
#APs_05: IoU = 0.5
#APs_all: IoU from 0.5-0.95 with increments of 0.05
image_ids = np.random.choice(dataset_val.image_ids, 100)
APs_05 = []
APs_all = []

cnt = 1
for image_id in image_ids:
    print("{}, {}/{}".format(image_id, cnt, len(image_ids)))
    # Load images and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask, gt_semantic, gt_depth = \
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP_05, precisions, recalls, overlaps = \
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs_05.append(AP_05)

    AP_all = \
        utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs_all.append(AP_all)
    cnt = cnt + 1

print("mAP: ", np.mean(APs_05))
print("mAP: ", np.mean(APs_all))





