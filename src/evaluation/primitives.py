from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class EvaluationExample:
    image: np.ndarray
    gt: np.ndarray
    original_example: dict
    example_id: str


@dataclass(frozen=True)
class InferenceResult:
    result: np.ndarray
    result_colors: np.ndarray
    inference_time: float


@dataclass(frozen=True)
class ClassEvaluation:
    dice: float
    iou: float
    pixels_voting: int

    def to_dict(self) -> dict:
        return {
            "dice": self.dice,
            "iou": self.iou,
            "pixels_voting": self.pixels_voting
        }


@dataclass(frozen=True)
class EvaluationResult:
    original_example: dict
    example_id: str
    class_based_metrics: Dict[str, ClassEvaluation]
    pixel_accuracy: float

    def to_dict(self) -> dict:
        return {
            "original_example": self.original_example,
            "example_id": self.example_id,
            "class_based_metrics": {
                key: value.to_dict()
                for key, value in self.class_based_metrics.items()
            },
            "pixel_accuracy": self.pixel_accuracy
        }


@dataclass(frozen=True)
class EvaluationResults:
    results_per_example: List[EvaluationResult]
    examples_weighted_class_metrics: Dict[str, Dict[str, Tuple[float, int]]]
    pixel_weighted_class_metrics: Dict[str, Dict[str, Tuple[float, int]]]
    examples_weighted_mean_metrics: Dict[str, float]
    pixel_weighted_mean_metrics: Dict[str, float]
    mean_pixel_accuracy: float
    mean_inference_time: float
    inference_time_variance: float

    def to_dict(self) -> dict:
        return {
            "results_per_example": [e.to_dict() for e in self.results_per_example],
            "examples_weighted_class_metrics": self.examples_weighted_class_metrics,
            "pixel_weighted_class_metrics": self.pixel_weighted_class_metrics,
            "examples_weighted_mean_metrics": self.examples_weighted_mean_metrics,
            "pixel_weighted_mean_metrics": self.pixel_weighted_mean_metrics,
            "mean_pixel_accuracy": self.mean_pixel_accuracy,
            "mean_inference_time": self.mean_inference_time,
            "inference_time_variance": self.inference_time_variance
        }
