import logging
import os
from glob import glob
from typing import List, Dict, Optional

from tqdm import tqdm

from src.config import ANNOTATIONS_DIR_NAME, LOGGING_LEVEL, MASK_EXTENSION, \
    MASK_NAME_POSTFIX
from src.preprocessing.annotations_parsing import AnnotationParser
from src.primitives.preprocessing import FileName2FilePath, \
    DataSetExamplePaths, DataSetExample
from src.utils.fs_utils import extract_file_name_without_extension
from src.utils.images import load_image

logging.getLogger().setLevel(LOGGING_LEVEL)


def fetch_images_with_annotations(dataset_path: str
                                  ) -> List[DataSetExample]:
    file_name2image = fetch_images_indexed_by_file_name(
        dataset_path=dataset_path
    )
    file_name2annotation = fetch_annotations_indexed_by_file_name(
        dataset_path=dataset_path
    )
    dataset_examples_paths = pair_image_with_annotation(
        file_name2image=file_name2image,
        file_name2annotation=file_name2annotation
    )
    return load_dataset_examples(data_set_examples_paths=dataset_examples_paths)


def fetch_images_indexed_by_file_name(dataset_path: str) -> FileName2FilePath:
    images_wildcard = os.path.join(dataset_path, "*", "*", "*.png")
    images = glob(images_wildcard)
    return convert_paths_into_file_names_based_lookup(
        file_paths=images
    )


def fetch_annotations_indexed_by_file_name(dataset_path: str
                                           ) -> FileName2FilePath:
    annotations_wild_card = os.path.join(
        dataset_path, "*", "*", "", ANNOTATIONS_DIR_NAME, "*.json"
    )
    annotations = glob(annotations_wild_card)
    return convert_paths_into_file_names_based_lookup(
        file_paths=annotations
    )


def convert_paths_into_file_names_based_lookup(file_paths: List[str]
                                               ) -> Dict[str, str]:
    lookup = {
        extract_file_name_without_extension(path): path for path in file_paths
    }
    if len(lookup) != len(file_paths):
        raise RuntimeError(
            "File names clash occurred between file names from different dirs."
        )
    return lookup


def pair_image_with_annotation(file_name2image: FileName2FilePath,
                               file_name2annotation: FileName2FilePath
                               ) -> List[DataSetExamplePaths]:
    images_file_names = set(file_name2image.keys())
    annotations_file_names = set(file_name2annotation.keys())
    common_file_names = images_file_names.intersection(annotations_file_names)
    return [
        DataSetExamplePaths(
            image_path=file_name2image[common_file_name],
            annotation_path=file_name2annotation[common_file_name]
        ) for common_file_name in common_file_names
    ]


def load_dataset_examples(data_set_examples_paths: List[DataSetExamplePaths]
                          ) -> List[DataSetExample]:
    dataset_examples = (
        convert_example_paths_into_example(example_paths)
        for example_paths in tqdm(data_set_examples_paths)
    )
    return [example for example in dataset_examples if example is not None]


def convert_example_paths_into_example(example_paths: DataSetExamplePaths
                                       ) -> Optional[DataSetExample]:
    annotation = AnnotationParser.parse_annotation(
        annotation_path=example_paths.annotation_path
    )
    image = load_image(image_path=example_paths.image_path)
    if image is None or annotation is None:
        return None
    dataset_example = DataSetExample(
        example_paths=example_paths,
        image=image,
        annotation=annotation
    )
    if dataset_example_invalid(dataset_example=dataset_example):
        logging.warning(
            f"Example {dataset_example} is invalid and will be rejected."
        )
        return None
    return dataset_example


def dataset_example_invalid(dataset_example: DataSetExample) -> bool:
    annotation_based_image_size = \
        dataset_example.annotation.image_size.to_compact_form()
    actual_image_size = dataset_example.image.shape[:2]
    constraints_failures = [
        annotation_based_image_size != actual_image_size,
        len(dataset_example.annotation.shapes) != 1
    ]
    return any(constraints_failures)


def create_path_for_mask(image_path: str) -> str:
    image_name = extract_file_name_without_extension(path=image_path)
    image_dir = os.path.dirname(image_path)
    return os.path.join(
        image_dir,
        f"{image_name}{MASK_NAME_POSTFIX}.{MASK_EXTENSION}"
    )
