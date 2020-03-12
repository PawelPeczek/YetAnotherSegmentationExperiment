from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.primitives.images import Shape, ImageSize

FileName2FilePath = Dict[str, str]


@dataclass(frozen=True)
class Annotation:
    name: str
    shapes: Dict[str, List[Shape]]
    image_size: ImageSize


@dataclass(frozen=True)
class DataSetExamplePaths:
    image_path: str
    annotation_path: str


@dataclass(frozen=True)
class DataSetExample:
    example_paths: DataSetExamplePaths
    image: np.ndarray
    annotation: Annotation
