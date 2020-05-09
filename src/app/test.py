import os

import requests
from matplotlib.patches import Patch

from data_access.data_transformations import DataTransformationChain
from data_access.folds_generation import FoldsGenerator
from primitives.data_access import DataSetSplit
from src.primitives.api import ServiceSpecs
import matplotlib.pyplot as plt

from data_access.data_generation import DataGenerator
from data_access.folds_generation import FoldsGenerator
import src.config as global_config
import src.data_access.config as data_access_config

import numpy as np
import cv2 as cv

from src.app.proxy import ObjectSegmentationServiceClient
from utils.api import image_to_jpeg_bytes
#
from utils.fs_utils import parse_json

EXPERIMENT_RESULTS_ROOT = "/home/mdronski/IET/semestr-8/UczenieMaszyn/YetAnotherSegmentationExperiment/resources/experiments/test_experiment_v1/ic_net/random_split_#0"
WEGIGHTS_PATH = os.path.join(
    EXPERIMENT_RESULTS_ROOT, "weights.hdf5"
)
SPLIT_JSON = os.path.join(
    EXPERIMENT_RESULTS_ROOT, "dataset.split.json"
)
SPLIT = parse_json(SPLIT_JSON)
SPLIT = DataSetSplit.from_dict(SPLIT)
test_generator = DataGenerator(
    examples=SPLIT.test_set.examples,
    transformation_chain=DataTransformationChain(
        entry_transformation=data_access_config.ENTRY_TRANSFORMATION,
        augmentations=[]
    ),
    batch_size=16
)
test_generator.on_epoch_end()
test_images, test_gt = test_generator[0]

# for i, image in enumerate(test_images):
#     np.save(str(i), image)

#
# img = np.load('15.npy')
# img_org = img
# # img = image_to_jpeg_bytes(img)
# # response = requests.post(
# #     'http://0.0.0.0:2137/v1/OBJECT_SEGMENTATION_SERVICE_NAME/object_segmentation', files={'image': img}, verify=False
# # )
#
servSpecs = ServiceSpecs(
    host='0.0.0.0',
    port=2137,
    service_name="OBJECT_SEGMENTATION_SERVICE_NAME",
    version='v1'
)

cli = ObjectSegmentationServiceClient(servSpecs)

legend_labels = ['backgroud'] + list(global_config.CLASS_MAPPINGS.keys())
legend_colors = list(global_config.CLASS_TO_COLORS)
legend = []
for label, color in zip(legend_labels, legend_colors):
    legend.append(Patch(color=color/255, label=label))

for img in test_images:
    # img = np.load(f"{i}.npy")
    results = cli.get_colorful_segmentation(img)
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(results.class_mask)
    plt.legend(handles=legend, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    # for result in results:
    #     class_idx = result.class_idx
    #     mask = result.binary_mask
    #     plt.subplot(121)
    #     plt.imshow(img)
    #     plt.subplot(122)
    #     plt.imshow(mask)
    #     plt.show()
