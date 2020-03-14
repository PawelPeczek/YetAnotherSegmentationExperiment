import logging
import random
from typing import List, Optional

import numpy as np
import cv2 as cv

from src.primitives.images import Shape, ImageSize, BinaryMask


MIN_CROP_DIM_FRACTION = 0.1


def load_image(image_path: str,
               color_conversion_flag: Optional[int] = None
               ) -> np.ndarray:
    image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
    if color_conversion_flag is not None:
        return cv.cvtColor(image, color_conversion_flag)
    else:
        return image


def create_mask_image(shapes: List[Shape], size: ImageSize) -> BinaryMask:
    mask = np.zeros(size.to_compact_form(), dtype=np.uint8)
    points = [
        [p.to_compact_form() for p in shape]
        for shape in shapes
    ]
    points = np.array(points, np.int32)
    points = points.squeeze(axis=0)
    cv.fillPoly(
        img=mask,
        pts=[points],
        color=255,
    )
    return mask.astype(np.bool)


def blend_mask_with_image(image: np.ndarray, mask: BinaryMask) -> np.ndarray:
    if image.shape[2] > 3:
        logging.warning("Image channels are trimmed to 3.")
        image = image[:, :, :3]
    mask = convert_binary_mask_into_image(mask=mask)
    return add_channel_to_image(
        image=image,
        channel=mask
    )


def convert_binary_mask_into_image(mask: BinaryMask) -> np.ndarray:
    return mask.astype(np.uint8) * 255


def add_alpha_channel_to_image(image: np.ndarray) -> np.ndarray:
    if image.shape[2] > 3:
        raise ValueError("Image already has more than 3 color channels.")
    alpha_channel = np.ones_like(image[:, :, 0]) * 255
    return add_channel_to_image(image=image, channel=alpha_channel)


def add_channel_to_image(image: np.ndarray, channel: np.ndarray) -> np.ndarray:
    if channel_shape_invalid(image=image, channel=channel):
        raise ValueError("Channel slice shape is invalid")
    if len(channel.shape) == 2:
        channel = np.expand_dims(channel, axis=-1)
    channel = channel.astype(np.uint8)
    return np.concatenate((image, channel), axis=-1)


def channel_shape_invalid(image: np.ndarray, channel: np.ndarray) -> bool:
    return len(channel.shape) not in {2, 3} or \
        channel.shape[:2] != image.shape[:2] or \
        channel.dtype != np.uint8


def blend_image_with_background(image: np.ndarray,
                                background: np.ndarray,
                                mask: BinaryMask,
                                background_random_adjust: bool = False
                                ) -> np.ndarray:
    background = adjust_background_to_image(
        background=background,
        image=image,
        background_random_adjust=background_random_adjust
    )
    return (image * mask + (1 - mask) * background).astype(np.uint8)


def adjust_background_to_image(background: np.ndarray,
                               image: np.ndarray,
                               background_random_adjust: bool
                               ) -> np.ndarray:
    if background_random_adjust is True:
        background = take_random_crop(image=background)
    return cv.resize(background, (image.shape[:2])[::-1])


def take_random_crop(image: np.ndarray) -> np.ndarray:
    min_height = int(round(MIN_CROP_DIM_FRACTION * image.shape[0]))
    crop_height = random.randint(min_height, image.shape[0])
    min_width = int(round(MIN_CROP_DIM_FRACTION * image.shape[0]))
    crop_width = random.randint(min_width, image.shape[1])
    left_top_y = random.randint(0, image.shape[0] - crop_height)
    left_top_x = random.randint(0, image.shape[1] - crop_width)
    return image[
           left_top_y:left_top_y + crop_height,
           left_top_x:left_top_x + crop_width
    ]
