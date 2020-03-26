import random
from typing import List, Tuple

import numpy as np
from tensorflow.python.keras.utils import Sequence

from src.data_access.data_transformations import DataTransformationChain
from src.primitives.data_access import DataSetExampleDescription


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
