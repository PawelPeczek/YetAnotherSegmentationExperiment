import os

from src.config import RESOURCES_PATH
from src.primitives.api import ServiceSpecs

API_VERSION = 'v1'
SERVICE_NAME = 'OBJECT_SEGMENTATION_SERVICE_NAME'
SERVICE_SECRET = 'OBJECT_SEGMENTATION_SERVICE_SECRET'
IDENTITY_SERVICE_SPECS = ServiceSpecs(
    host='SERVER_IDENTITY_SERVICE_HOST',
    port=2137,
    service_name='SERVER_IDENTITY_SERVICE_NAME',
    version=API_VERSION
)
DISCOVERY_SERVICE_SPECS = ServiceSpecs(
    host='DISCOVERY_SERVICE_HOST',
    port=6666,
    service_name='DISCOVERY_SERVICE_NAME',
    version=API_VERSION
)
VERIFY_SERVICE_IDENTITY_PATH = 'verify_service_identity'
LOCATE_SERVICES_PATH = 'locate_services'
SEGMENTATION_ENDPOINT = 'object_segmentation'
# WEIGHTS_PATH = os.path.join(RESOURCES_PATH, "weights", "weights.hdf5")
WEIGHTS_PATH = "/home/ppeczek/Dokumenty/YetAnotherSegmentationExperiment/" \
               "resources/experiments/test_experiment_v1/ic_net/random_split_#0/" \
               "weights.hdf5"

CONFIDENCE_THRESHOLD = 0.8
