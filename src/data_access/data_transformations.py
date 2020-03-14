from abc import ABC, abstractmethod
from typing import List, Dict, Optional

import numpy as np
import cv2 as cv

from src.primitives.data_access import DataSetExample, \
    DataSetExampleDescription, DataSetExampleBatch
from src.primitives.images import ImageSize, DimensionsOrder
from src.utils.images import load_image, blend_image_with_background
from src.utils.iterables import random_sample, parallel_map, split_list


class DataTransformation(ABC):
    pass


class EntryTransformation(DataTransformation):

    def __init__(self,
                 class_mapping: Dict[str, int],
                 target_size: ImageSize,
                 backgrounds: List[np.ndarray],
                 workers: int = 8):
        self.__class_mapping = class_mapping
        self.__target_size = target_size
        self.__backgrounds = backgrounds
        self.__workers = workers

    def transform_batch(self,
                        example_descriptions: List[DataSetExampleDescription],
                        ) -> DataSetExampleBatch:
        batch = parallel_map(
            iterable=example_descriptions,
            map_function=self.transform_example,
            workers_number=self.__workers
        )
        images, ground_truths = split_list(to_split=batch)
        images_batch = np.stack(images, axis=0)
        ground_truths_batch = np.stack(ground_truths, axis=0)
        return images_batch, ground_truths_batch

    def transform_example(self,
                          example_description: DataSetExampleDescription
                          ) -> DataSetExample:
        image_rgba = load_image(
            image_path=example_description.image_path,
            color_conversion_flag=cv.COLOR_BGRA2RGBA
        )
        image, mask = image_rgba[:, :, :3], image_rgba[:, :, -1:]
        mask = mask.astype(np.bool)
        image = self.__blend_image_with_background(
            image=image,
            mask=mask,
            background_path=example_description.fixed_background_path
        )
        class_id = self.__class_mapping[example_description.class_name]
        mask = mask.astype(np.uint8) * class_id
        return self.__adjust_dataset_example_scale(
            image=image,
            mask=mask
        )

    def __blend_image_with_background(self,
                                      image: np.ndarray,
                                      mask: np.ndarray,
                                      background_path: Optional[str]
                                      ) -> np.ndarray:
        background_not_fixed = background_path is None
        if background_not_fixed:
            background = random_sample(to_sample=self.__backgrounds)
        else:
            background = load_image(
                image_path=background_path,
                color_conversion_flag=cv.COLOR_BGR2RGB
            )
        return blend_image_with_background(
            image=image,
            background=background,
            mask=mask,
            background_random_adjust=background_not_fixed
        )

    def __adjust_dataset_example_scale(self,
                                       image: np.ndarray,
                                       mask: np.ndarray
                                       ) -> DataSetExample:
        target_size = self.__target_size.to_compact_form(
            dimensions_order=DimensionsOrder.WIDTH_HEIGHT
        )
        if image.shape[:2] != target_size:
            image = cv.resize(image, target_size)
            mask = cv.resize(mask, target_size, interpolation=cv.INTER_NEAREST)
        return image, mask


class DataAugmentation(DataTransformation):

    @abstractmethod
    def transform_batch(self,
                        batch: DataSetExampleBatch
                        ) -> DataSetExampleBatch:
        pass

    @abstractmethod
    def transform_example(self,
                          dataset_example: DataSetExample
                          ) -> DataSetExample:
        pass


class DataTransformationChain(DataTransformation):

    def __init__(self,
                 entry_transformation: EntryTransformation,
                 augmentations: List[DataAugmentation]):
        self.__chain = [entry_transformation]
        self.__chain.extend(augmentations)

    def transform_batch(self,
                        example_descriptions: List[DataSetExampleDescription],
                        ) -> DataSetExampleBatch:
        results = example_descriptions
        for transformation in self.__chain:
            results = transformation.transform_batch(results)
        return results

    def transform_example(self,
                          example_description: DataSetExampleDescription
                          ) -> DataSetExample:
        results = example_description
        for transformation in self.__chain:
            results = transformation.transform_example(results)
        return results
