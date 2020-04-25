from glob import glob
import cv2 as cv
from albumentations import (
    HorizontalFlip, VerticalFlip, Blur, RandomGamma, Rotate, ShiftScaleRotate,
    OpticalDistortion, GridDistortion, ElasticTransform, HueSaturationValue,
    RGBShift, MotionBlur, GaussianBlur, CLAHE, ChannelShuffle, ToGray,
    Downscale, FancyPCA, Posterize, Equalize, ISONoise, RandomFog,
    RandomBrightnessContrast, ImageCompression, CoarseDropout
)

from src.config import CLASS_MAPPINGS, BACKGROUNDS_WILDRCARD, MODEL_INPUT_SIZE
from src.data_access.data_transformations import EntryTransformation, \
    DataAugmentation, DataStandardisation, DataTransformationChain
from src.primitives.data_access import RandomSplitSpecs, \
    RotationBasedClassSplit, FoldsGeneratorSpecs
from src.utils.images import load_image


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
    HorizontalFlip(p=0.1),
    VerticalFlip(p=0.1),
    Blur(blur_limit=16, p=0.1),
    RandomGamma(gamma_limit=(60, 140), p=0.1),
    Rotate(limit=35, p=0.15),
    ShiftScaleRotate(rotate_limit=35, p=0.2),
    OpticalDistortion(distort_limit=1.0, shift_limit=1.0, p=0.2),
    HueSaturationValue(
        hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.2
    ),
    RGBShift(
        r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.15
    ),
    RandomBrightnessContrast(p=0.2),
    MotionBlur(blur_limit=7, p=0.2),
    GaussianBlur(blur_limit=7, p=0.15),
    CLAHE(p=0.05),
    ChannelShuffle(p=0.05),
    ToGray(p=0.1),
    ImageCompression(quality_lower=10, quality_upper=100, p=0.15),
    CoarseDropout(max_holes=32, max_height=12, max_width=12, p=0.05),
    Downscale(p=0.3),
    FancyPCA(alpha=0.4, p=0.1),
    Posterize(num_bits=4, p=0.03),
    Equalize(p=0.05),
    ISONoise(color_shift=(0.1, 0.5), p=0.07),
    RandomFog(p=0.03)
]

BACKGROUNDS_PATHS = glob(BACKGROUNDS_WILDRCARD)
BACKGROUNDS = [load_image(path, cv.COLOR_BGR2RGB) for path in BACKGROUNDS_PATHS]

ENTRY_TRANSFORMATION = EntryTransformation(
    class_mapping=CLASS_MAPPINGS,
    target_size=MODEL_INPUT_SIZE,
    backgrounds=BACKGROUNDS
)

DATA_AUGMENTATIONS = [
    DataAugmentation(
        transformations=AUGMENTATIONS,
        global_application_probab=0.6
    ),
    DataStandardisation()
]

TRAINING_TRANSFORMATION_CHAIN = DataTransformationChain(
    entry_transformation=ENTRY_TRANSFORMATION,
    augmentations=DATA_AUGMENTATIONS
)
VALIDATION_TRANSFORMATION_CHAIN = DataTransformationChain(
    entry_transformation=ENTRY_TRANSFORMATION,
    augmentations=[DataStandardisation()]
)
