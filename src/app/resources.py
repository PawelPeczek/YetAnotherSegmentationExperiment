import logging
import time
from typing import Tuple

from albumentations import ToFloat, Normalize
from flask import Response, request, make_response
from flask_restful import Resource
import numpy as np
import cv2 as cv
from tensorflow.python.keras import Model
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

from data_access.config import VALIDATION_TRANSFORMATION_CHAIN
from data_access.data_transformations import DataStandardisation
from src.primitives.segmentation_result import \
    SegmentationResult, SegmentationResults, ColorfulSegmentationResult
from src.utils.api import image_from_str, map_color

logging.getLogger().setLevel(logging.INFO)


def standarize_image(image):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    tf = ToFloat(max_value=255.0)
    norm = Normalize(mean=mean, std=std, max_pixel_value=1.0)
    image = tf(image=image)['image']
    image = norm(image=image)['image']
    return image


class ObjectSegmentation(Resource):

    def __init__(self,
                 model: Model,
                 session: tf.Session,
                 graph: tf.Graph,
                 confidence_threshold: float,
                 max_image_dim: int = 128):
        self.__model = model
        self.__session = session
        self.__graph = graph
        self.__confidence_threshold = confidence_threshold
        self.__max_image_dim = max_image_dim

    def post(self) -> Response:
        if 'image' not in request.files:
            return make_response({'msg': 'Field named "image" required.'}, 500)
        image = image_from_str(raw_image=request.files['image'].read())
        image = standarize_image(image)
        start_time = time.time()
        results = self.__infer_from_image(image=image)
        logging.info(f"INFERENCE TIME: {time.time() - start_time}s")
        detected_objects = self.__post_process_inference(results)
        return make_response({'objects': detected_objects}, 200)

    def __infer_from_image(self,
                           image: np.ndarray
                           ) -> np.ndarray:
        # image, scale = self.__standardize_image(image=image)
        # logging.info(f"Standardized image shape: {image.shape}. scale: {scale}")
        expanded_image = np.expand_dims(image, axis=0)
        set_session(self.__session)
        with self.__graph.as_default():
            results = self.__model.predict_on_batch(
                x=expanded_image
            )
            return results

    def __post_process_inference(self, results: np.ndarray) -> SegmentationResults:
        inference_results = []
        for result in results:
            single_inference_results = []
            for cls_idx in range(result.shape[-1]):
                if cls_idx == 0:
                    continue
                class_result = result[..., cls_idx]
                mask = class_result > self.__confidence_threshold
                if np.any(mask):
                    single_inference_results.append(
                        SegmentationResult(cls_idx, mask).to_dict()
                    )
            inference_results.append(single_inference_results)

        return inference_results


class ColorfulObjectSegmentation(Resource):
    def __init__(self,
                 model: Model,
                 session: tf.Session,
                 graph: tf.Graph,
                 confidence_threshold: float,
                 max_image_dim: int = 128):
        self.__model = model
        self.__session = session
        self.__graph = graph
        self.__confidence_threshold = confidence_threshold
        self.__max_image_dim = max_image_dim

    def post(self) -> Response:
        if 'image' not in request.files:
            return make_response({'msg': 'Field named "image" required.'}, 500)
        image = image_from_str(raw_image=request.files['image'].read())
        image = standarize_image(image)
        start_time = time.time()
        results = self.__infer_from_image(image=image)
        logging.info(f"INFERENCE TIME: {time.time() - start_time}s")
        detected_objects = self.__post_process_inference(results)
        return make_response({'objects': detected_objects}, 200)

    def __infer_from_image(self,
                           image: np.ndarray
                           ) -> np.ndarray:
        expanded_image = np.expand_dims(image, axis=0)
        set_session(self.__session)
        with self.__graph.as_default():
            results = self.__model.predict_on_batch(
                x=expanded_image
            )
            class_mask = np.argmax(np.squeeze(results, axis=0), axis=-1)
            return map_color(class_map=class_mask)

    def __post_process_inference(self, results: np.ndarray) -> dict:
        return ColorfulSegmentationResult(class_mask=results).to_dict()
