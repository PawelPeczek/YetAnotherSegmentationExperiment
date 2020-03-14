from abc import ABC, abstractmethod
from functools import partial
from typing import Set, Dict, List

from src.data_access.utils import get_dataset_images_by_classes, \
    split_list_by_given_factor, get_dataset_images_by_classes_and_angles, \
    create_dataset_descriptions, calculate_classes_balance, \
    assign_backgrounds_to_examples
from src.primitives.data_access import DataSetSplit, ExamplesPathsSplit, \
    RotationBasedClassSplit, DataSetSplitDescription, ExamplesPathsByClass, \
    DataSetSplitGenerator, RandomSplitSpecs
from src.utils.iterables import split_dictionary_by_value, \
    extract_and_merge_dictionary_sub_groups, flatten_dictionary_of_lists


class SplitGenerator(ABC):

    def __init__(self, dataset_path: str):
        self._dataset_path = dataset_path

    @abstractmethod
    def generate_splits(self) -> DataSetSplitGenerator:
        pass

    def _prepare_dataset_split(self,
                               name: str,
                               train_examples: ExamplesPathsByClass,
                               test_examples: ExamplesPathsByClass
                               ) -> DataSetSplit:
        training_set = SplitDescriptionCreator.create_split_description(
            samples_by_class=train_examples
        )
        test_set = SplitDescriptionCreator.create_split_description(
            samples_by_class=test_examples
        )
        test_set = self.__fix_test_examples_backgrounds(test_split=test_set)
        return DataSetSplit(
            name=name,
            training_set=training_set,
            test_set=test_set
        )

    def __fix_test_examples_backgrounds(self,
                                        test_split: DataSetSplitDescription
                                        ) -> DataSetSplitDescription:
        examples_with_fixed_backgrounds = assign_backgrounds_to_examples(
            examples=test_split.examples,
            data_set_dir=self._dataset_path
        )
        return DataSetSplitDescription(
            examples=examples_with_fixed_backgrounds,
            classes_balance=test_split.classes_balance
        )


class RandomSplitGenerator(SplitGenerator):

    def __init__(self,
                 dataset_path: str,
                 random_split_specs: RandomSplitSpecs):
        super().__init__(dataset_path=dataset_path)
        self.__random_split_specs = random_split_specs

    def generate_splits(self) -> DataSetSplitGenerator:
        for index in range(self.__random_split_specs.splits_number):
            yield self.__prepare_split(index=index)

    def __prepare_split(self, index: int) -> DataSetSplit:
        train_examples, test_examples = self.__split_dataset()
        return self._prepare_dataset_split(
            name=f"Random split #{index}",
            train_examples=train_examples,
            test_examples=test_examples
        )

    def __split_dataset(self) -> ExamplesPathsSplit:
        images_by_classes = get_dataset_images_by_classes(
            dataset_path=self._dataset_path,
            classes=self.__random_split_specs.classes
        )
        split_list = partial(
            split_list_by_given_factor,
            factor=-self.__random_split_specs.training_samples_factor
        )
        dataset_split = {
            class_name: split_list(class_examples)
            for class_name, class_examples in images_by_classes.items()
        }
        return split_dictionary_by_value(
            dictionary=dataset_split
        )


class RotationBasedSplitGenerator(SplitGenerator):

    def __init__(self,
                 dataset_path: str,
                 splits_specs: List[RotationBasedClassSplit]):
        super().__init__(dataset_path=dataset_path)
        self.__splits_specs = splits_specs

    def generate_splits(self) -> DataSetSplitGenerator:
        for index, split_specs in enumerate(self.__splits_specs):
            yield self.__prepare_split(index=index, split_specs=split_specs)

    def __prepare_split(self,
                        index: int,
                        split_specs: RotationBasedClassSplit
                        ) -> DataSetSplit:
        training_set_paths, test_set_paths = self.__split_dataset(
            split_specs=split_specs
        )
        return self._prepare_dataset_split(
            name=f"Rotation-based split #{index}",
            train_examples=training_set_paths,
            test_examples=test_set_paths
        )

    def __split_dataset(self,
                        split_specs: RotationBasedClassSplit
                        ) -> ExamplesPathsSplit:
        classes = self.__induce_classes_from_split_specs(
            split_specs=split_specs
        )
        grouped_dataset = get_dataset_images_by_classes_and_angles(
            dataset_path=self._dataset_path,
            classes=classes
        )
        training_set_paths = extract_and_merge_dictionary_sub_groups(
            dictionary=grouped_dataset,
            to_extract=split_specs.training_angles
        )
        test_set_paths = extract_and_merge_dictionary_sub_groups(
            dictionary=grouped_dataset,
            to_extract=split_specs.training_angles
        )
        return training_set_paths, test_set_paths

    def __induce_classes_from_split_specs(self,
                                          split_specs: RotationBasedClassSplit
                                          ) -> Set[str]:
        training_classes = set(split_specs.training_angles.keys())
        test_classes = set(split_specs.test_angles.keys())
        return training_classes.union(test_classes)


class SplitDescriptionCreator:

    @staticmethod
    def create_split_description(samples_by_class: Dict[str, List[str]]
                                 ) -> DataSetSplitDescription:
        dataset_examples_by_class = {
            class_name: create_dataset_descriptions(example_paths, class_name)
            for class_name, example_paths in samples_by_class.items()
        }
        classes_balance = calculate_classes_balance(
            examples_by_class=dataset_examples_by_class
        )
        examples = flatten_dictionary_of_lists(
            dictionary=dataset_examples_by_class
        )
        return DataSetSplitDescription(
            examples=examples,
            classes_balance=classes_balance
        )
