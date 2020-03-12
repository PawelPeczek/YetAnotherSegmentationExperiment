import cv2 as cv

from src.preprocessing.utils import fetch_images_with_annotations, \
    create_path_for_mask
from src.primitives.preprocessing import DataSetExample
from src.utils.images import create_mask_image, blend_mask_with_image, \
    convert_binary_mask_into_image
from src.utils.iterables import for_each


class AnnotationsConverter:

    @staticmethod
    def convert_all_images(dataset_root_path: str) -> None:
        dataset_examples = fetch_images_with_annotations(
            dataset_path=dataset_root_path
        )
        for_each(
            iterable=dataset_examples,
            side_effect=AnnotationsConverter.__convert_single_example
        )

    @staticmethod
    def __convert_single_example(dataset_example: DataSetExample) -> None:
        shapes_class = list(dataset_example.annotation.shapes.keys())[0]
        shapes = dataset_example.annotation.shapes[shapes_class]
        mask = create_mask_image(
            shapes=shapes,
            size=dataset_example.annotation.image_size
        )
        converted_image = blend_mask_with_image(
            image=dataset_example.image,
            mask=mask
        )
        image_path = dataset_example.example_paths.image_path
        cv.imwrite(image_path, converted_image)
        converted_mask = convert_binary_mask_into_image(mask=mask)
        mask_path = create_path_for_mask(image_path=image_path)
        cv.imwrite(mask_path, converted_mask)
