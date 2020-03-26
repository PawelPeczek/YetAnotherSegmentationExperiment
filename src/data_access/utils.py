import math
import os
from functools import partial, reduce
from typing import Set, Dict, List, Optional, TypeVar, Tuple

from glob import glob
import numpy as np

from src.config import MASK_EXTENSION, MASK_NAME_POSTFIX, \
    BACKGROUNDS_DIR_NAME, BACKGROUND_CLASS
from src.primitives.data_access import DataSetExampleDescription, \
    ExampleDescriptionsByClass, ExamplesPathsByClass, \
    ExamplesPathsByClassAndAngle
from src.utils.fs_utils import make_path_relative
from src.utils.iterables import append_to_dictionary_of_lists, \
    count_dictionary_values, sum_dictionary_values, add_grouping_to_dictionary, \
    random_sample, trim_dictionary_values, standardise_dictionary_values
from src.utils.numbers import safe_cast_str2int

V = TypeVar('V')


def get_dataset_images_by_classes_and_angles(dataset_path: str,
                                             classes: Set[str]
                                             ) -> ExamplesPathsByClassAndAngle:
    image_paths_by_classes = get_dataset_images_by_classes(
        dataset_path=dataset_path,
        classes=classes
    )
    angle_value = partial(
        extract_angle_from_example_path, dataset_path=dataset_path
    )
    return add_grouping_to_dictionary(
        dictionary=image_paths_by_classes,
        group_by=angle_value
    )


def get_dataset_images_by_classes(dataset_path: str,
                                  classes: Set[str]
                                  ) -> ExamplesPathsByClass:
    images_wildcard = os.path.join(dataset_path, '*', '*', '*.png')
    all_images = glob(images_wildcard)
    images_without_masks = filter_out_masks_from_images_paths(
        images_paths=all_images
    )
    extract_class = partial(
        extract_class_name_from_example_path,
        dataset_path=dataset_path
    )
    image_classes = (extract_class(p) for p in images_without_masks)
    dict_specs = (
        (class_name, path) 
        for class_name, path in zip(image_classes, images_without_masks)
        if class_name is not None and class_name in classes
    )
    return reduce(append_to_dictionary_of_lists, dict_specs, {})


def filter_out_masks_from_images_paths(images_paths: List[str]) -> List[str]:
    mast_postfix = f'{MASK_NAME_POSTFIX}.{MASK_EXTENSION}'
    return [p for p in images_paths if not p.endswith(mast_postfix)]


def extract_class_name_from_example_path(example_path: str,
                                         dataset_path: str
                                         ) -> Optional[str]:
    relative_example_path = make_path_relative(
        path=example_path,
        reference_path=dataset_path
    )
    if relative_example_path is None:
        return None
    relative_example_path_chunks = relative_example_path.split('/')
    if len(relative_example_path_chunks) != 3:
        return None
    return relative_example_path_chunks[0]


def extract_angle_from_example_path(example_path: str,
                                    dataset_path: str
                                    ) -> Optional[int]:
    relative_example_path = make_path_relative(
        path=example_path,
        reference_path=dataset_path
    )
    if relative_example_path is None:
        return None
    relative_example_path_chunks = relative_example_path.split('/')
    if len(relative_example_path_chunks) != 3:
        return None
    return safe_cast_str2int(relative_example_path_chunks[1])


def split_list_by_given_factor(input_list: List[V],
                               factor: float
                               ) -> Tuple[List[V], List[V]]:
    if factor < 0.0 or factor > 1.0:
        raise ValueError("Factor must be in range [0;1]")
    split_position = int(math.floor(len(input_list) * factor))
    return input_list[:split_position], input_list[split_position:]


def create_dataset_descriptions(example_paths: List[str],
                                class_name: str
                                ) -> List[DataSetExampleDescription]:
    return [
        DataSetExampleDescription(image_path=path, class_name=class_name)
        for path in example_paths
    ]


def calculate_classes_balance(examples_by_class: ExampleDescriptionsByClass
                              ) -> Dict[str, float]:
    examples_number_by_class = count_dictionary_values(
        dictionary=examples_by_class
    )
    all_examples_number = sum_dictionary_values(
        dictionary=examples_number_by_class
    )
    return {
        key: value / all_examples_number
        for key, value in examples_number_by_class.items()
    }


def assign_backgrounds_to_examples(examples: List[DataSetExampleDescription],
                                   data_set_dir: str
                                   ) -> List[DataSetExampleDescription]:
    return [
        assign_background_to_example(example, data_set_dir)
        for example in examples
    ]


def assign_background_to_example(example: DataSetExampleDescription,
                                 data_set_dir: str
                                 ) -> DataSetExampleDescription:
    background_path = sample_background_path(data_set_dir=data_set_dir)
    return DataSetExampleDescription(
        image_path=example.image_path,
        class_name=example.class_name,
        fixed_background_path=background_path
    )


def sample_background_path(data_set_dir: str) -> str:
    background_wildcard = os.path.join(
        data_set_dir, BACKGROUNDS_DIR_NAME, "*.jpg"
    )
    backgrounds = glob(background_wildcard)
    if len(backgrounds) == 0:
        raise RuntimeError("Cannot find backgrounds.")
    return random_sample(to_sample=backgrounds)


def prepare_class_weighting(class_balance: Dict[str, float],
                            class_mapping: Dict[str, int],
                            stability_coefficient: float = 1e-7
                            ) -> Dict[int, float]:
    classes_names_intersection = set(class_balance.keys()).intersection(
        set(class_mapping.keys())
    )
    classes_names_union = set(class_balance.keys()).union(
        set(class_mapping.keys())
    )
    if classes_names_intersection != classes_names_union:
        raise ValueError(
            "Class balance dict is not compliant with class mapping."
        )
    class_weighting = {}
    for key in class_balance:
        class_id = class_mapping[key]
        class_weighting[class_id] = \
            1.0 / (class_balance[key] + stability_coefficient)
    return _standardise_class_weighting(class_weighting=class_weighting)


def _standardise_class_weighting(class_weighting: Dict[int, float]
                                 ) -> Dict[int, float]:
    class_weighting[BACKGROUND_CLASS] = \
        min(class_weighting.values()) / len(class_weighting)
    maximum_coefficient = (6 * len(class_weighting))
    class_weighting = trim_dictionary_values(
        dictionary=class_weighting,
        max_value=maximum_coefficient
    )
    return standardise_dictionary_values(dictionary=class_weighting)


def convert_class_weighting_to_vector(class_weighting: Dict[int, float]
                                      ) -> np.ndarray:
    classes_idx = (class_weighting.keys())
    result = []
    for i in range(max(classes_idx) + 1):
        result.append(class_weighting.get(i, 0.0))
    return np.array(result, dtype=np.float32)
