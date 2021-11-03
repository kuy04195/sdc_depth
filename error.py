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
import json

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
IMAGE_DIR = os.path.join(ROOT_DIR, "../data/leftImg8bit/train/")


class CityscapeConfig(Config):
    """Configuration for training on the cityscape dataset.
    Derives from the base Config class and overrides values specific
    to the cityscape dataset.
    """
    NAME = "cityscape"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1  # 8

    STEPS_PER_EPOCH = 100#6000

    VALIDATION_STEPS = 20#300

    BACKBONE = "resnet50"

    NUM_CLASSES = 1 + 15  # background + 1 shapes

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024

    LEARNING_RATE = 0.01

class InferenceConfig(CityscapeConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

inference_config = InferenceConfig()
#model = modellib.MaskRCNN(mode="inference",
#                          config=inference_config,
#                          model_dir=MODEL_DIR)

#model_path = model.find_last()
#model.load_weights(model_path, by_name=True)

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

pic_cnt = 0
NaN_cul = 0
def load_depth(image_dir):
    depth_dir = image_dir.replace("leftImg8bit", "disparity", 1).replace("leftImg8bit", "disparity")
    depth_map = skimage.io.imread(depth_dir)

    #print(np.sum(depth_map == 1), np.sum(np.logical_and(depth_map > 1, depth_map < 10)))

    camera_dir = image_dir.replace("leftImg8bit", "camera", 1).replace("leftImg8bit", "camera").replace(".png", ".json")
    with open(camera_dir) as f:
        camera = json.load(f)
    baseline        = camera['extrinsic']['baseline']
    focal_length    = camera['intrinsic']['fx']

    depth_map = (depth_map.astype(np.float32) - 1.) / 256.
    NaN = depth_map <= 0
    zeros = depth_map < 0
    #depth_map[NaN] = -1
    #depth_map       = baseline * focal_length / depth_map
    #depth_map[NaN] = 0

    global pic_cnt
    global NaN_cul
    pic_cnt += 1.
    NaN_cul += np.sum(NaN).astype(np.float64)
    print(NaN_cul / (1024*2048) * 100 / pic_cnt)
    """train : 19.15, val : 20.16%"""
    #plt.plasma()
    #plt.imshow(depth_map)
    #plt.savefig('save.png')
    #exit(0)
    
    return depth_map

def error(_rmse, _rmse_log, _abs, _sq):
    print("rmse     =", _rmse,      end='\t')
    print("rmse_log =", _rmse_log,  end='\t')
    print("abs      =", _abs,       end='\t')
    print("sq       =", _sq,        end='\n')

epsilon = 1e-6
RMSE = []
RMSE_LOG = []
ABS = []
SQ = []

for image_name in file_names:
    image_dir = os.path.join(IMAGE_DIR, image_name)
    image = skimage.io.imread(image_dir)
    gt_depth = load_depth(image_dir)

    # Run detection
    #results = model.detect([image], verbose=0)
    #r = results[0]

    nan_cnt = np.sum(gt_depth == 0)

    pixels = np.size(gt_depth) - nan_cnt
    """
    prediction = r['depth']
    NaN = prediction == 0
    prediction[NaN] = -1
    prediction = baseline * focal_length / prediction
    prediction[NaN] = 0

    prediction = np.multiply(prediction, gt_depth != 0)
    gt_depth[gt_depth==0] = epsilon
    prediction[prediction==0] = epsilon

    _rmse = np.sqrt(np.sum((prediction - gt_depth)**2) / pixels)
    _rmse_log = np.sqrt(np.sum((np.log(prediction) - np.log(gt_depth))**2) / pixels)
    _abs = np.sum(np.abs(prediction - gt_depth)) / pixels
    _sq = np.sum(prediction**2 - gt_depth**2) / pixels
    error(_rmse, _rmse_log, _abs, _sq)
    RMSE.append(_rmse)
    RMSE_LOG.append(_rmse_log)
    ABS.append(_abs)
    SQ.append(_sq)"""

print('RMSE      =', np.mean(RMSE))
print('RMSE(log) =', np.mean(RMSE_LOG))
print('Abs Rel   =', np.mean(ABS))
print('Sq Rel    =', np.mean(SQ))