import logging
import os
import numpy as np

from src.primitives.images import ImageSize

LOGGING_LEVEL = logging.INFO
RESOURCES_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'resources'
))
MASK_NAME_POSTFIX = "_mask"
MASK_EXTENSION = "png"
DATASET_PATH = os.path.join(RESOURCES_PATH, "VISAPP_extended_dataset")
EXPERIMENTS_OUTPUT_DIR = os.path.join(RESOURCES_PATH, "experiments")
ANNOTATIONS_DIR_NAME = 'annotations'
BACKGROUNDS_DIR_NAME = 'backgrounds'
BACKGROUNDS_WILDRCARD = os.path.join(DATASET_PATH, BACKGROUNDS_DIR_NAME, "*.jpg")
GOOGLE_DRIVE_RESOURCE_ID = '18q3xisLUsP-LXwACFTdaiwltA-cCBLgq'
BACKGROUND_CLASS = 0
BACKGROUND_CLASS_NAME = "background"
CLASS_MAPPINGS = {
    "adapter": 1,
    "bottle": 2,
    "box": 3,
    "clamp": 4,
    "drill": 5,
    "duck": 6
}
CLASS_MAPPINGS_REVERTED = {
    1: "adapter",
    2: "bottle",
    3: "box",
    4: "clamp",
    5: "drill",
    6: "duck"
}
CLASS_TO_COLORS = np.array([
    (127, 127, 127),
    (255, 0, 0),
    (0, 255, 0),
    (255, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (249, 215, 28)
])
MODEL_INPUT_SIZE = ImageSize(
    height=128,
    width=128
)
