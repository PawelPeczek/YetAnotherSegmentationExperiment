from glob import glob
import cv2 as cv
from albumentations import (
    HorizontalFlip, VerticalFlip, Blur, RandomGamma, Rotate, ShiftScaleRotate,
    OpticalDistortion, GridDistortion, ElasticTransform, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, MotionBlur, GaussianBlur,
    CLAHE, ChannelShuffle, ToGray, JpegCompression, Cutout, Downscale,
    FancyPCA, Posterize, Equalize, ISONoise, RandomFog
)

from config import CLASS_MAPPINGS, BACKGROUNDS_WILDRCARD
from data_access.data_transformations import EntryTransformation, DataAugmentation, DataStandardisation, \
    DataTransformationChain
from primitives.data_access import RandomSplitSpecs, RotationBasedClassSplit, FoldsGeneratorSpecs
from primitives.images import ImageSize
from utils.images import load_image

BATCH_SIZE = 32

RANDOM_SPLIT_SPECS = RandomSplitSpecs(
    classes=set(CLASS_MAPPINGS.keys()),
    splits_number=1,
    training_samples_factor=0.8
)

TRAINING_ANGLES = {
    "adapter": {0, 30, 90},
    "bottle": {0, 60},
    "box": {0, 60},
    "clamp": {0, 60},
    "drill": {0, 30, 90},
    "duck": {0, 60}
}

TEST_ANGLES = {
    "adapter": {60},
    "bottle": {30},
    "box": {30},
    "clamp": {30},
    "drill": {60},
    "duck": {30}
}

ROTATION_BASED_CLASS_SPLIT = RotationBasedClassSplit(
    training_angles=TRAINING_ANGLES,
    test_angles=TEST_ANGLES
)

FOLDS_GENERATOR_SPECS = FoldsGeneratorSpecs(
    random_split_specs=RANDOM_SPLIT_SPECS,
    rotation_based_splits=[ROTATION_BASED_CLASS_SPLIT]
)

AUGMENTATIONS = [
    HorizontalFlip(always_apply=False, p=0.15),
    VerticalFlip(always_apply=False, p=0.15),
    Blur(blur_limit=16, always_apply=False, p=0.15),
    RandomGamma(gamma_limit=(60, 140), always_apply=False, p=0.1),
    Rotate(limit=35, always_apply=False, p=0.15),
    ShiftScaleRotate(rotate_limit=35, always_apply=False, p=0.1),
    OpticalDistortion(distort_limit=1.0, shift_limit=1.0, always_apply=False, p=0.1),
    GridDistortion(always_apply=False, p=0.1),
    ElasticTransform(always_apply=False, p=0.1),
    HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.1),
    RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.1),
    RandomBrightness(limit=0.3, always_apply=False, p=0.1),
    RandomContrast(limit=0.3, always_apply=False, p=0.1),
    MotionBlur(blur_limit=7, always_apply=False, p=0.1),
    GaussianBlur(blur_limit=7, always_apply=False, p=0.1),
    CLAHE(always_apply=False, p=0.1),
    ChannelShuffle(always_apply=False, p=0.05),
    ToGray(always_apply=False, p=0.5),
    JpegCompression(quality_lower=10, quality_upper=100, always_apply=False, p=0.15),
    Cutout(num_holes=32, max_h_size=12, max_w_size=12, always_apply=False, p=0.07),
    Downscale(always_apply=False, p=0.2),
    FancyPCA(alpha=0.4, always_apply=False, p=0.1),
    Posterize(num_bits=4, always_apply=False, p=0.03),
    Equalize(always_apply=False, p=0.03),
    ISONoise(color_shift=(0.1, 0.5), always_apply=False, p=0.07),
    RandomFog(always_apply=False, p=0.03),
]

TARGET_SIZE_ELEMENT = ImageSize(
    height=480,
    width=640
)

backgrounds_paths = glob(BACKGROUNDS_WILDRCARD)
BACKGROUNDS = [load_image(path, cv.COLOR_BGR2RGB) for path in backgrounds_paths]

ENTRY_TRANSFORMATION = EntryTransformation(
    class_mapping=CLASS_MAPPINGS,
    target_size=TARGET_SIZE_ELEMENT,
    backgrounds=BACKGROUNDS
)

DATA_AUGMENTATIONS = [
    DataAugmentation(
        transformations=AUGMENTATIONS,
        global_application_probab=0.7
    ),
    DataStandardisation()
]

TRANSFORMATION_CHAIN = DataTransformationChain(
    entry_transformation=ENTRY_TRANSFORMATION,
    augmentations=DATA_AUGMENTATIONS
)
