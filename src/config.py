import logging
import os

LOGGING_LEVEL = logging.INFO
RESOURCES_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'resources'
))
MASK_NAME_POSTFIX = "_mask"
MASK_EXTENSION = "png"
DATASET_PATH = os.path.join(RESOURCES_PATH, "VISAPP_extended_dataset")
ANNOTATIONS_DIR_NAME = 'annotations'
BACKGROUNDS_DIR_NAME = 'backgrounds'
BACKGROUNDS_WILDRCARD = os.path.join(DATASET_PATH, BACKGROUNDS_DIR_NAME, "*.jpg")
GOOGLE_DRIVE_RESOURCE_ID = '17LG2ed6pfZQPlq5SkiOFl7djuhmls2No'
CLASS_MAPPINGS = {
    "adapter": 1,
    "bottle": 2,
    "box": 3,
    "clamp": 4,
    "drill": 5,
    "duck": 6
}
