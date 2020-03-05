import os
from dataclasses import dataclass
from functools import reduce
from typing import Dict, Optional, Tuple

from src.preprocessing.config import SHAPES_KEY, LABEL_KEY, HEIGHT_KEY, \
    WIDTH_KEY, POINTS_KEY, IMAGE_PATH_KEY
from src.primitives.files import ParsedJSON
from src.primitives.images import Shape, ImageSize, Point
from src.utils.fs_utils import safe_parse_json, \
    extract_file_name_without_extension
from src.utils.iterables import append_to_dictionary_of_lists


@dataclass(frozen=True)
class Annotation:
    name: str
    shapes: Dict[str, Shape]
    image_size: ImageSize


class AnnotationParser:

    @staticmethod
    def parse_annotation(annotation_path: str) -> Optional[Annotation]:
        parsed_json_annotation = safe_parse_json(path=annotation_path)
        if parsed_json_annotation is None:
            return None
        image_size = AnnotationParser.__parse_image_size(
            parsed_json_annotation=parsed_json_annotation
        )
        image_path = AnnotationParser.__parse_image_path(
            parsed_json_annotation=parsed_json_annotation,
            annotation_path=annotation_path
        )
        if image_size is None or image_path is None:
            return None
        name = extract_file_name_without_extension(path=annotation_path)
        raw_shapes = parsed_json_annotation.get(SHAPES_KEY)
        shapes = map(AnnotationParser.__parse_shape, raw_shapes)
        shapes = list(filter(lambda shape: shape is not None, shapes))
        if len(shapes) == 0:
            return None
        shapes = reduce(append_to_dictionary_of_lists, shapes, {})
        return Annotation(
            name=name,
            shapes=shapes,
            image_size=image_size
        )

    @staticmethod
    def __parse_image_path(parsed_json_annotation: ParsedJSON,
                           annotation_path: str
                           ) -> Optional[str]:
        image_path = parsed_json_annotation.get(IMAGE_PATH_KEY)
        if image_path is None:
            return None
        return os.path.abspath(os.path.join(
            annotation_path, image_path
        ))

    @staticmethod
    def __parse_shape(raw_shape: dict) -> Optional[Tuple[str, Shape]]:
        label = raw_shape.get(LABEL_KEY)
        if label is None:
            return None
        points = raw_shape.get(POINTS_KEY, [])
        points = map(Point.from_compact_form, points)
        points = list(filter(lambda p: p is not None, points))
        if len(points) == 0:
            return None
        return label, points

    @staticmethod
    def __parse_image_size(parsed_json_annotation: ParsedJSON
                           ) -> Optional[ImageSize]:
        height = parsed_json_annotation.get(HEIGHT_KEY)
        width = parsed_json_annotation.get(WIDTH_KEY)
        if any(e is None for e in [height, width]):
            return None
        return ImageSize(
            height=height,
            width=width
        )
