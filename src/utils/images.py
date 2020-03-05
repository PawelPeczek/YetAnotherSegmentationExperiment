from typing import List

import numpy as np
import cv2 as cv

from src.primitives.images import Shape, ImageSize, BinaryMask


def create_mask_image(shapes: List[Shape], size: ImageSize) -> BinaryMask:
    mask = np.zeros(size.to_compact_form(), dtype=np.uint8)
    points = [
        [p.to_compact_form() for p in shape]
        for shape in shapes
    ]
    points = np.array(points, np.int32)
    points = points.reshape((-1, 1, 2))
    cv.polylines(
        img=mask,
        pts=points,
        isClosed=True,
        color=255,
        thickness=-1
    )
    return mask.astype(np.bool)
