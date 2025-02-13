import random
from typing import List, Tuple

import numpy as np
from tensorflow.python.keras.utils import Sequence

from src.data_access.data_transformations import DataTransformationChain
from src.primitives.data_access import DataSetExampleDescription, DataSetExampleBatch


class DataGenerator(Sequence):
    def __init__(self,
                 examples: List[DataSetExampleDescription],
                 transformation_chain: DataTransformationChain,
                 batch_size: int = 32):
        super().__init__()
        self.__examples = examples
        self.__transformation_chain = transformation_chain
        self.__batch_size = batch_size

    def on_epoch_end(self) -> None:
        random.shuffle(self.__examples)

    def __len__(self) -> int:
        return int(np.floor(len(self.__examples) / self.__batch_size))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        start_index, end_index = \
            index * self.__batch_size, (index + 1) * self.__batch_size
        batch_examples = self.__examples[start_index:end_index]
        return self.__transformation_chain.transform_batch(
            example_descriptions=batch_examples
        )


class DetailedDataGenerator(Sequence):
    def __init__(self,
                 data: List[Tuple[List[DataSetExampleDescription], DataTransformationChain]],
                 batch_size: int = 32):
        super().__init__()
        self.__data = data
        self.__batch_size = batch_size

    def on_epoch_end(self) -> None:
        self.__examples = self.__prepare_examples()
        e, t = self.__examples
        l = list(zip(e, t))
        random.shuffle(l)
        self.__examples = l

    def __len__(self) -> int:
        return int(np.floor(len(self.__examples) / self.__batch_size))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        start_index, end_index = index * self.__batch_size, (index + 1) * self.__batch_size
        return self.__examples[start_index:end_index]

    def __prepare_examples(self) -> DataSetExampleBatch:
        examples_truths = []
        for image_paths, transformation_chain in self.__data:
            examples_truths.append(transformation_chain.transform_batch(image_paths))

        all_batches = examples_truths[0][0]
        all_truths = examples_truths[0][1]
        for example_batch, ground_truths in examples_truths[1:]:
            all_batches = np.concatenate((all_batches, example_batch))
            all_truths = np.concatenate((all_truths, ground_truths))

        return all_batches, all_truths