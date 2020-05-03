from typing import List

import requests
from flask import Response

from src.primitives.api import ServiceSpecs
from primitives.segmentation_result import SegmentationResult, SegmentationResults
from src.utils.api import image_to_jpeg_bytes, compose_url, compose_url_with_http_prefix
from src.app.config import SEGMENTATION_ENDPOINT
from src.app.errors import RequestProcessingError


import numpy as np


class ObjectSegmentationServiceClient:

    def __init__(self,
                 object_segmentation_service_specs: ServiceSpecs):
        self.__object_segmentation_service_specs = object_segmentation_service_specs

    def get_segmentation(self, image: np.ndarray) -> SegmentationResults:
        raw_image = image_to_jpeg_bytes(image=image)
        files = {'image': raw_image}
        url = compose_url_with_http_prefix(
            service_specs=self.__object_segmentation_service_specs,
            path_postfix=SEGMENTATION_ENDPOINT
        )
        response = requests.post(
            url, files=files, verify=False
        )
        if response.status_code == 200:
            return self.__convert_results(response)
        else:
            raise RequestProcessingError(
                f'Error code: {response.status_code}, Cause: {response.text}'
            )

    def __convert_results(self, response: Response) -> SegmentationResults:
        segmentation_results = []
        objects = response.json()['objects']
        for result in objects:
            for class_result in result:
                segmentation_results.append(SegmentationResult.from_dict(class_result))
        return segmentation_results
