from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, List, Optional, Any

import numpy as np


class CoordinatesOrder(Enum):
    XY = 0
    YX = 1


RawCompactPoint = List[Any]
CompactPoint = Tuple[int, int]


@dataclass(frozen=True)
class Point:
    x: int
    y: int

    def to_compact_form(self,
                        coordinates_order: CoordinatesOrder = CoordinatesOrder.XY
                        ) -> CompactPoint:
        if coordinates_order is CoordinatesOrder.XY:
            compact_point = self.x, self.y
        else:
            compact_point = self.y, self.x
        return compact_point

    @classmethod
    def from_compact_form(cls,
                          compact_form: RawCompactPoint,
                          coordinates_order: CoordinatesOrder = CoordinatesOrder.XY
                          ) -> Optional[Point]:
        try:
            first_coord, second_coord = \
                int(round(compact_form[0])), int(round(compact_form[1]))
        except Exception:
            return None
        if coordinates_order is CoordinatesOrder.XY:
            return cls(
                x=first_coord,
                y=second_coord
            )
        else:
            return cls(
                x=second_coord,
                y=first_coord
            )


Shape = List[Point]


class DimensionsOrder(Enum):
    HEIGHT_WIDTH = 0
    WIDTH_HEIGHT = 1


CompactImageSize = Tuple[int, int]


@dataclass(frozen=True)
class ImageSize:
    height: int
    width: int

    def to_compact_form(self,
                        dimensions_order: DimensionsOrder = DimensionsOrder.HEIGHT_WIDTH
                        ) -> CompactImageSize:
        if dimensions_order is DimensionsOrder.HEIGHT_WIDTH:
            compact_image_size = self.height, self.width
        else:
            compact_image_size = self.width, self.height
        return compact_image_size


BinaryMask = np.ndarray
