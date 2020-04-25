from __future__ import annotations
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
    allowed_background_wildcard: str = None


@dataclass(frozen=True)
class DataSetExampleDescription:
    image_path: str
    class_name: str
    fixed_background_path: Optional[str] = None

    @classmethod
    def from_dict(cls,
                  description: Dict[str, Optional[str]]
                  ) -> DataSetExampleDescription:
        return cls(
            image_path=description["image_path"],
            class_name=description["class_name"],
            fixed_background_path=description.get("fixed_background_path")
        )

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "image_path": self.image_path,
            "class_name": self.class_name,
            "fixed_background_path": self.fixed_background_path
        }


@dataclass(frozen=True)
class DataSetSplitDescription:
    examples: List[DataSetExampleDescription]
    classes_balance: Dict[str, float]

    @classmethod
    def from_dict(cls,
                  description: dict
                  ) -> DataSetSplitDescription:
        examples = [
            DataSetExampleDescription.from_dict(e)
            for e in description.get("examples", [])
        ]
        return cls(
            examples=examples,
            classes_balance=description["classes_balance"]
        )

    def to_dict(self) -> dict:
        return {
            "examples": [e.to_dict() for e in self.examples],
            "classes_balance": self.classes_balance
        }


@dataclass(frozen=True)
class DataSetSplit:
    name: str
    training_set: DataSetSplitDescription
    test_set: DataSetSplitDescription

    @classmethod
    def from_dict(cls,
                  split: dict
                  ) -> DataSetSplit:
        return cls(
            name=split["name"],
            training_set=DataSetSplitDescription.from_dict(split["training_set"]),
            test_set=DataSetSplitDescription.from_dict(split["test_set"])
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "training_set": self.training_set.to_dict(),
            "test_set": self.test_set.to_dict()
        }


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
