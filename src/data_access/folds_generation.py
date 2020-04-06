from typing import Generator

from src.data_access.split_generation import RandomSplitGenerator, \
    RotationBasedSplitGenerator
from src.primitives.data_access import FoldsGeneratorSpecs, DataSetSplit


class FoldsGenerator:

    def __init__(self,
                 dataset_path: str,
                 generator_specs: FoldsGeneratorSpecs):
        self.__random_split_generator = RandomSplitGenerator(
            dataset_path=dataset_path,
            random_split_specs=generator_specs.random_split_specs
        )
        self.__random_split_generator.set_backround_wildcard(generator_specs.allowed_background_wildcard)

        self.__rotation_based_split_generator = RotationBasedSplitGenerator(
            dataset_path=dataset_path,
            splits_specs=generator_specs.rotation_based_splits
        )
        self.__rotation_based_split_generator.set_backround_wildcard(generator_specs.allowed_background_wildcard)

    def generate_folds(self) -> Generator[DataSetSplit, None, None]:
        for split in self.__random_split_generator.generate_splits():
            yield split
        for split in self.__rotation_based_split_generator.generate_splits():
            yield split
