import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import skimage
import matplotlib
import matplotlib.pyplot as plt
import PIL

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "../data/leftImg8bit/val/")

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
    STEPS_PER_EPOCH = 100#6000

    # Number of validation steRPNps to run at the end of every training epoch.
    VALIDATION_STEPS = 20#300

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 15  # background + 1 shapes

    # Input image resing
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024

    # Learning rate and momentum
    LEARNING_RATE = 0.01

config = CityscapeConfig()
class InferenceConfig(CityscapeConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

inference_config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

model_path = model.find_last()
model.load_weights(model_path, by_name=True)

def load_semantic(image_dir):
    semantic_dir = image_dir.replace("leftImg8bit", "gtFine", 1)[:-len("_leftImg8bit.png")]+"_semantic"
    semantic_list = os.listdir(semantic_dir)

    semantic = np.zeros([1024, 2048, config.NUM_CLASSES])
    for sem in semantic_list:
        category_id = int(sem[:-len(".png")])
        tmp_sem_path = os.path.join(semantic_dir, sem)
        tmp_semantic = 255 - skimage.io.imread(tmp_sem_path)
        semantic[:, :, category_id] = tmp_semantic
    return semantic

class_names = ["BG", "bicycle", "motorcycle", "train", "bus", "truck", "car", "person", "sky", "vegetation",
    "traffic sign", "traffic light", "building", "sidewalk", "road", "ground"]

file_names = []
cities = os.listdir(IMAGE_DIR)
for city in cities:
    city_path = os.path.join(IMAGE_DIR, city)
    images = os.listdir(city_path)
    for img in images:
        img_dir = os.path.join(city_path, img)
        file_names.append(img_dir)
random_files = np.random.choice(file_names, 4)

SQ = []
cnt = 1
for random_file in random_files:
    image_dir = os.path.join(IMAGE_DIR, random_file)
    image = skimage.io.imread(image_dir)

    # Run detection
    results = model.detect([image], verbose=0)
    r = results[0]

    ground_truth = load_semantic(image_dir)
    prediction = r['semantic']

    IoU = []
    for i in range(config.NUM_CLASSES):
        gt = ground_truth[:,:,i] > 0.5
        p = prediction[:,:,i]
        
        if np.sum(gt) == 0:
            IoU.append(-1)
            continue
        Union = np.logical_or(gt, p)
        Intersection = np.logical_and(gt, p)
        IoU.append(np.sum(Intersection) / np.sum(Union))

    IoU = np.array(IoU)
    SQ.append(IoU)
    valid = (IoU >= 0)

    print(cnt, end=' ')
    if np.any(valid) == False:
        print("SQ : 0")
    else :
        sq = np.mean(IoU[valid])
        print("SQ :", sq, "|", IoU.astype(np.float32))
    cnt = cnt + 1

SQ = np.array(SQ)
valid = (SQ >= 0)
SQ[SQ < 0] = 0

SQ_per_class = np.sum(SQ, axis=0)
SQ_per_image = np.sum(SQ, axis=1)

valid_per_class = np.sum(valid, axis=0)
valid_per_image = np.sum(valid, axis=1)

for i in range(config.NUM_CLASSES):
    if valid_per_class[i] == 0:
        print(class_names[i], ": doesn't exist")
        continue
    print(class_names[i], ":", SQ_per_class[i] / valid_per_class[i])

print("mSQ : ", np.sum(SQ) / np.sum(valid))
print(valid_per_class)
print(valid_per_image)