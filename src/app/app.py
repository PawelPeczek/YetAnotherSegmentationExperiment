from flask import Flask
from flask_jwt_extended import JWTManager
from flask_restful import Api
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

import src.config as global_config
from models.icnet import ICNet
from src.app.resources import ObjectSegmentation, ColorfulObjectSegmentation
from src.utils.api import compose_relative_resource_url
from src.app.config import SERVICE_NAME, API_VERSION, \
    WEIGHTS_PATH, CONFIDENCE_THRESHOLD, SEGMENTATION_ENDPOINT, COLORFUL_SEGMENTATION_ENDPOINT

from tensorflow.python.keras.utils.generic_utils import get_custom_objects
from src.evaluation.losses.dice import dice_loss, bce_dice_loss

get_custom_objects().update({"bce_dice_loss": bce_dice_loss})
get_custom_objects().update({"dice_loss": dice_loss})


def build_icnet_model():
    ic_net = ICNet(num_classes=len(global_config.CLASS_MAPPINGS) + 1)
    input_shape = global_config.MODEL_INPUT_SIZE.to_compact_form() + (3, )
    model = ic_net.build_model(input_shape=input_shape)
    model.load_weights(WEIGHTS_PATH)
    return model


INTER_SERVICES_TOKEN = None
app = Flask(__name__)
GRAPH = None
TF_SESSION = None
app.config['PROPAGATE_EXCEPTIONS'] = True
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
jwt = JWTManager(app)


def create_api() -> Api:
    global INTER_SERVICES_TOKEN
    api = Api(app)
    global TF_SESSION
    global GRAPH
    GRAPH = tf.get_default_graph()
    TF_SESSION = tf.Session(graph=GRAPH)
    set_session(TF_SESSION)
    model = build_icnet_model()
    api.add_resource(
        ObjectSegmentation,
        compose_relative_resource_url(SERVICE_NAME, API_VERSION, SEGMENTATION_ENDPOINT),
        resource_class_kwargs={
            'model': model,
            'session': TF_SESSION,
            'graph': GRAPH,
            'confidence_threshold': CONFIDENCE_THRESHOLD,
            'max_image_dim': 128,
        }
    )
    api.add_resource(
        ColorfulObjectSegmentation,
        compose_relative_resource_url(SERVICE_NAME, API_VERSION, COLORFUL_SEGMENTATION_ENDPOINT),
        resource_class_kwargs={
            'model': model,
            'session': TF_SESSION,
            'graph': GRAPH,
            'confidence_threshold': CONFIDENCE_THRESHOLD,
            'max_image_dim': 128,
        }
    )
    return api


api = create_api()


if __name__ == '__main__':
    port = 2137
    app.run(host='0.0.0.0', port=port)
