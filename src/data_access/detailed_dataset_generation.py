import os
from glob import glob
from typing import List, Tuple

import cv2 as cv

from src.config import DATASET_PATH, BACKGROUNDS_DIR_NAME, MODEL_INPUT_SIZE
from src.data_access.config import TRAINING_ANGLES, TEST_ANGLES, AUGMENTATIONS
from src.data_access.data_transformations import EntryTransformation, DataTransformationChain, DataStandardisation, \
    DataAugmentation
from src.data_access.folds_generation import FoldsGenerator
from src.primitives.data_access import RandomSplitSpecs, RotationBasedClassSplit, FoldsGeneratorSpecs, \
    DataSetExampleDescription
from src.utils.images import load_image


class DetailedDatasetHandler:
    def __init__(self,
                 classes: List[str],
                 difficulty: str) -> None:
        self._classes = classes
        self._difficulty = difficulty

    def __generate_backgrounds_wildcard(self, clazz: str) -> str:
        class_background_path_wildcard = {
            "all": "*/",
            "easy": "easy/",
            "hard": "blending/"
        }[self._difficulty]
        return os.path.join(DATASET_PATH, BACKGROUNDS_DIR_NAME, clazz, class_background_path_wildcard, "*.jpg")

    def __generate_paths(self, clazz: str):
        wildcard = self.__generate_backgrounds_wildcard(clazz)
        return glob(wildcard)

    def __generate_transformation_chain(self, clazz: str, standarize: bool = False, augument: bool = False) -> DataTransformationChain:
        backgrounds_paths = self.__generate_paths(clazz)
        backgrounds = [load_image(path, cv.COLOR_BGR2RGB) for path in backgrounds_paths]

        entry_transformation = EntryTransformation(
            class_mapping={clazz: 1},
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

    def generate_folds_generator(self, clazz: str) -> FoldsGenerator:
        random_split_specs = RandomSplitSpecs(
            classes={clazz},
            splits_number=1,
            training_samples_factor=0.8
        )

        rotation_based_class_split = RotationBasedClassSplit(
            training_angles={clazz: TRAINING_ANGLES[clazz]},
            test_angles={clazz: TEST_ANGLES[clazz]}
        )

        folds_generator_specs = FoldsGeneratorSpecs(
            random_split_specs=random_split_specs,
            rotation_based_splits=[rotation_based_class_split],
            allowed_background_wildcard=self.__generate_backgrounds_wildcard(clazz)
        )

        return FoldsGenerator(
            dataset_path=DATASET_PATH,
            generator_specs=folds_generator_specs
        )

    def prepare_generators_and_transformations(self) -> List[Tuple[FoldsGenerator, DataTransformationChain]]:
        return [
            (self.generate_folds_generator(clazz), self.__generate_transformation_chain(clazz))
            for clazz in self._classes
        ]

    def prepare_folds_and_transformations(self, standarize: bool = False, augument: bool = False) -> List[Tuple[List[DataSetExampleDescription], DataTransformationChain]]:
        return [
            (next(self.generate_folds_generator(clazz).generate_folds()).training_set.examples,
             self.__generate_transformation_chain(clazz, standarize, augument))
            for clazz in self._classes
        ]

