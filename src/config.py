import logging
import os

from src.primitives.images import ImageSize

LOGGING_LEVEL = logging.INFO
RESOURCES_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'resources'
))
MASK_NAME_POSTFIX = "_mask"
MASK_EXTENSION = "png"
DATASET_PATH = os.path.join(RESOURCES_PATH, "VISAPP_extended_dataset")
MODELS_PATH = os.path.join(RESOURCES_PATH, "models")
ANNOTATIONS_DIR_NAME = 'annotations'
BACKGROUNDS_DIR_NAME = 'backgrounds'
BACKGROUNDS_WILDRCARD = os.path.join(DATASET_PATH, BACKGROUNDS_DIR_NAME, "*.jpg")
HARD_BACKGROUNDS_WILDRCARD = os.path.join(DATASET_PATH, BACKGROUNDS_DIR_NAME, "*", "blending", "*.jpg")
EASY_BACKGROUNDS_WILDRCARD = os.path.join(DATASET_PATH, BACKGROUNDS_DIR_NAME, "*", "easy", "*.jpg")
GOOGLE_DRIVE_RESOURCE_ID = '18q3xisLUsP-LXwACFTdaiwltA-cCBLgq'
BACKGROUND_CLASS = 0
CLASS_MAPPINGS = {
    "adapter": 1,
    "bottle": 2,
    "box": 3,
    "clamp": 4,
    "drill": 5,
    "duck": 6
}
MODEL_INPUT_SIZE = ImageSize(
    height=128,
    width=128
)
