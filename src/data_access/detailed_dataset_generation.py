import os
from glob import glob

from src.data_access.config import TRAINING_ANGLES, TEST_ANGLES, AUGMENTATIONS
from src.data_access.folds_generation import FoldsGenerator
from src.primitives.data_access import RandomSplitSpecs, RotationBasedClassSplit, FoldsGeneratorSpecs
from src.config import DATASET_PATH, BACKGROUNDS_DIR_NAME, CLASS_MAPPINGS, MODEL_INPUT_SIZE
from src.data_access.data_transformations import EntryTransformation, DataTransformationChain, DataStandardisation, \
    DataAugmentation
from src.utils.images import load_image
import cv2 as cv


class DetailedDatasetHandler:
    def __init__(self,
                 clazz: str,
                 difficulty: str) -> None:
        self._clazz = clazz
        self._difficulty = difficulty

    def generate_backgrounds_wildcard(self):
        class_background_path_wildcard = {
            "all": "*/",
            "easy": "easy/",
            "hard": "blending/"
        }[self._difficulty]
        return os.path.join(DATASET_PATH, BACKGROUNDS_DIR_NAME, self._clazz, class_background_path_wildcard, "*.jpg")

    def generate_paths(self):
        wildcard = self.generate_backgrounds_wildcard()
        return glob(wildcard)

    def generate_transformation_chain(self, standarize: bool = False, augument: bool = False):
        backgrounds_paths = self.generate_paths()
        backgrounds = [load_image(path, cv.COLOR_BGR2RGB) for path in backgrounds_paths]

        entry_transformation = EntryTransformation(
            class_mapping={self._clazz: 1},
            target_size=MODEL_INPUT_SIZE,
            backgrounds=backgrounds
        )

        augumentations = []

        if augument:
            augumentations.append(
                DataAugmentation(
                    transformations=AUGMENTATIONS,
                    global_application_probab=0.6
                ))

        if standarize:
            augumentations.append(DataStandardisation())

        return DataTransformationChain(
            entry_transformation=entry_transformation,
            augmentations=augumentations
        )

    def generate_folds_generator(self,):
        random_split_specs = RandomSplitSpecs(
            classes={self._clazz},
            splits_number=1,
            training_samples_factor=0.8
        )

        rotation_based_class_split = RotationBasedClassSplit(
            training_angles={self._clazz: TRAINING_ANGLES[self._clazz]},
            test_angles={self._clazz: TEST_ANGLES[self._clazz]}
        )

        folds_generator_specs = FoldsGeneratorSpecs(
            random_split_specs=random_split_specs,
            rotation_based_splits=[rotation_based_class_split],
            allowed_background_wildcard=self.generate_backgrounds_wildcard()
        )

        return FoldsGenerator(
            dataset_path=DATASET_PATH,
            generator_specs=folds_generator_specs
        )
