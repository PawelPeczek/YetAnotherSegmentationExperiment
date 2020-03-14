from dataclasses import dataclass
from typing import Set, List, Dict, Optional, Tuple, Generator
import numpy as np

ClassToAnglesOfChoice = Dict[str, Set[int]]


@dataclass(frozen=True)
class RotationBasedClassSplit:
    training_angles: ClassToAnglesOfChoice
    test_angles: ClassToAnglesOfChoice


@dataclass(frozen=True)
class RandomSplitSpecs:
    classes: Set[str]
    splits_number: int
    training_samples_factor: float


@dataclass(frozen=True)
class FoldsGeneratorSpecs:
    random_split_specs: RandomSplitSpecs
    rotation_based_splits: List[RotationBasedClassSplit]


@dataclass(frozen=True)
class DataSetExampleDescription:
    image_path: str
    class_name: str
    fixed_background_path: Optional[str] = None


@dataclass(frozen=True)
class DataSetSplitDescription:
    examples: List[DataSetExampleDescription]
    classes_balance: Dict[str, float]


@dataclass(frozen=True)
class DataSetSplit:
    name: str
    training_set: DataSetSplitDescription
    test_set: DataSetSplitDescription


ExamplesPathsByClass = Dict[str, List[str]]
ExamplesPathsByClassAndAngle = Dict[str, Dict[int, List[str]]]
ExamplesPathsSplit = Tuple[ExamplesPathsByClass, ExamplesPathsByClass]
ExampleDescriptionsByClass = Dict[str, List[DataSetExampleDescription]]
DataSetSplitGenerator = Generator[DataSetSplit, None, None]


Example = np.ndarray
# Image (H, W, 3)
GroundTruth = np.ndarray
# Ground truth binary mask (H, W) with values {0, 1, ..., class number}
DataSetExample = Tuple[Example, GroundTruth]

ExamplesBatch = np.ndarray
# Batch of images of size (N, H, W, 3)
GroundTruthsBatch = np.ndarray
# Batch of GroundTruths of size (N, H, W)
DataSetExampleBatch = Tuple[ExamplesBatch, GroundTruthsBatch]
