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

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()

# Load trained weights (fill in path to trained weights here)
# assert model_path != "", "Provide path to trained weights"
# print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)
# model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = ["BG", "bicycle", "motorcycle", "train", "bus", "truck", "car", "person", "sky", "vegetation",
    "traffic sign", "traffic light", "building", "sidewalk", "road", "ground"]

# Load random images from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]

file_names = []
cities = os.listdir(IMAGE_DIR)
for city in cities:
    city_path = os.path.join(IMAGE_DIR, city)
    images = os.listdir(city_path)
    for img in images:
        img_dir = os.path.join(city_path, img)
        file_names.append(img_dir)
        
"""
for idx, img_dir in enumerate(file_names):
    image = skimage.io.imread(os.path.join(IMAGE_DIR, img_dir))
    nan = np.isinf(image)
    assert np.any(nan) == False, img_dir + "has nan"
    print(idx, img_dir, "is ok")"""

random_files = np.random.choice(file_names, 2)
for random_file in random_files:
    image = skimage.io.imread(os.path.join(IMAGE_DIR, random_file))

    # Run detection
    results = model.detect([image], verbose=1)
    r = results[0]

    # Visualize results
    print(random_file)
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])
    
    visualize.display_semantics(image, r['semantic'], class_names)
    
    plt.clf()
    plt.cla()
    plt.plasma()
    depth_plot = plt.imshow(r['depth'])
    d = r['depth']
    print(np.max(d), np.min(d), np.mean(d), np.std(d))
    #plt.savefig(os.path.basename(random_file).replace("leftImg8bit", "depth"))
    skimage.io.imsave("./{}".format(os.path.basename(random_file)), r['depth'])