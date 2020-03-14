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
GOOGLE_DRIVE_RESOURCE_ID = '17LG2ed6pfZQPlq5SkiOFl7djuhmls2No'
CLASS_MAPPINGS = {
    "adapter": 0,
    "bottle": 1,
    "box": 2,
    "clamp": 3,
    "drill": 4,
    "duck": 5
}
