from abc import abstractmethod
from typing import Tuple

from tensorflow.python.keras.engine.training import Model


class SegmentationModel:

    def __init__(self, num_classes: int):
        self._num_classes = num_classes

    @abstractmethod
    def build_model(self, input_shape: Tuple[int, int, int]) -> Model:
        pass
