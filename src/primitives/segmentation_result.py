import json
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

@dataclass
class SegmentationResult:
    class_idx: int
    binary_mask: np.ndarray

    def to_dict(self) -> Dict:
        return {
            'class_idx': self.class_idx,
            'mask': self.binary_mask.tolist()
        }

    @classmethod
    def from_dict(cls, segmentation_result_dict):
        class_idx = segmentation_result_dict['class_idx']
        mask = segmentation_result_dict['mask']
        return cls(
            class_idx=class_idx,
            binary_mask=mask
        )


SegmentationResults = List[SegmentationResult]