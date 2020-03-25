import random
from abc import ABC, abstractmethod

from typing import List, Generator

from keras.engine.training_utils import make_batches
from keras.utils import Sequence

from config import DATASET_PATH
from data_access.config import FOLDS_GENERATOR_SPECS, TRANSFORMATION_CHAIN, BATCH_SIZE
from data_access.folds_generation import FoldsGenerator
from primitives.data_access import DataSetExampleDescription, DataSetSplit


class DataGenerator(ABC, Sequence):
    def __init__(self) -> None:
        super().__init__()
        self.splits = self.__create_splits()
        self.batch_count = self.__calculate_batch_count()
        self.on_epoch_end()
        self.indexes = self.prepare_batch_indexes()

    def __create_splits(self) -> Generator[DataSetSplit, None, None]:
        return FoldsGenerator(
            dataset_path=DATASET_PATH,
            generator_specs=FOLDS_GENERATOR_SPECS
        ).generate_folds()

    def on_epoch_end(self):
        split = next(self.splits)
        all_batch = self.get_data(split)
        random.shuffle(all_batch)
        self.data = TRANSFORMATION_CHAIN.transform_batch(all_batch)

    def __calculate_batch_count(self):
        gen = self.__create_splits()
        dataset_size = len(next(gen).training_set.examples)
        return len(make_batches(dataset_size, BATCH_SIZE))

    @abstractmethod
    def get_data(self, split: DataSetSplit) -> List[DataSetExampleDescription]:
        pass

    def __len__(self):
        return self.batch_count-1

    def __getitem__(self, index):
        return self.__get_slice_from_batch(self.indexes[index])

    def prepare_batch_indexes(self):
        return make_batches(len(self.data[0]), BATCH_SIZE)

    def __get_slice_from_batch(self, batch_idx):
        start = batch_idx[0]
        end = batch_idx[1]
        X = self.data[0][start:end, :, :, :]
        y = self.data[1][start:end, :, :]
        return X, y


class TrainGenerator(DataGenerator):
    def get_data(self, split: DataSetSplit) -> List[DataSetExampleDescription]:
        return split.training_set.examples


class TestGenerator(DataGenerator):
    def get_data(self, split: DataSetSplit) -> List[DataSetExampleDescription]:
        return split.test_set.examples


