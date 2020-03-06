from os import listdir
from os.path import isfile, join

from preprocessing.config import ANGLES
from src.preprocessing.annotations_parsing import Annotation, AnnotationParser
from src.utils.images import create_mask_image
import cv2 as cv
import numpy as np
import os


class AnnotationsConverter:

    def __init__(self,
                 dataset_root_path: str,
                 output_sub_directories_name: str
                 ):
        self.__dataset_root_path = dataset_root_path
        self.__output_sub_directories_name = output_sub_directories_name

    def convert_all_images(self) -> None:
        for angle in ANGLES:
            self.__convert_angle_images(angle)

    def __convert_angle_images(self, angle: str) -> None:
        angle_path = join(self.__dataset_root_path, angle)
        image_paths = [f for f in listdir(angle_path) if isfile(join(angle_path, f))]

        for image_path in image_paths:
            self.__convert_single_image(os.path.join(angle_path, image_path))

    def __convert_single_image(self, image_path: str) -> None:
        img = cv.imread(image_path, cv.IMREAD_UNCHANGED)

        img_RGBA = self.__load_rgba_image(img)
        base_output_path = self.__create_base_output_path(image_path)

        mask = self.__create_mask(base_output_path)

        self.__save_mask_image(base_output_path, mask)
        self.__save_masked_rgba_image(base_output_path, img_RGBA, mask)

    def __load_rgba_image(self, img: np.ndarray) -> np.ndarray:
        r_channel, g_channel, b_channel = cv.split(img)
        alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255  # creating a dummy alpha channel image.
        img_RGBA = cv.merge((r_channel, g_channel, b_channel, alpha_channel))
        return img_RGBA

    def __create_base_output_path(self, image_path: str) -> str:
        image_name_without_extension = os.path.basename(image_path).split(sep=".")[0]
        curr_dir_path = os.path.dirname(image_path)
        output_path = os.path.join(curr_dir_path, self.__output_sub_directories_name)
        base_output_path = os.path.join(output_path, image_name_without_extension)
        return base_output_path

    def __create_mask(self, base_output_path: str) -> np.ndarray:
        annotation_path = base_output_path + '.json'
        annotation = AnnotationParser.parse_annotation(annotation_path)
        mask = create_mask_image([annotation.shapes['adaptator']], annotation.image_size)
        return mask

    def __save_mask_image(self, base_output_path: str, mask: np.ndarray) -> None:
        mask_path = base_output_path + '_mask.png'
        cv.imwrite(mask_path, mask * 255)

    def __save_masked_rgba_image(self, base_output_path: str, img_rgba: np.ndarray, mask: np.ndarray) -> None:
        img_rgba[mask, :4] = 255
        masked_image_path = base_output_path + '_masked_image.png'
        cv.imwrite(masked_image_path, img_rgba)
