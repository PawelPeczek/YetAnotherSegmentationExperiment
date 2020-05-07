import cv2
import os
import time
from copy import copy
from glob import glob
from typing import Dict, List, Tuple

from sklearn.metrics import confusion_matrix
from tensorflow.python.keras import Model
import numpy as np
from tqdm import tqdm
from tensorflow.python.keras import backend as K
from uuid import uuid4

from src.config import MODEL_INPUT_SIZE, CLASS_TO_COLORS, \
    CLASS_MAPPINGS_REVERTED, BACKGROUND_CLASS, BACKGROUND_CLASS_NAME
from src.data_access.config import VALIDATION_TRANSFORMATION_CHAIN
from src.data_access.data_generation import DataGenerator
from src.evaluation.config import EVALUATION_METRICS_FILE_NAME, \
    RESULTS_DIR_NAMES
from src.evaluation.primitives import EvaluationExample, InferenceResult, \
    EvaluationResults, EvaluationResult, ClassEvaluation
from src.models.base import SegmentationModel
from src.primitives.data_access import DataSetSplit
from src.training.config import SPLIT_SPECS_FILE_NAME, WEIGHTS_FILE_NAME
from src.utils.fs_utils import parse_json, dump_json
from src.utils.iterables import append_to_dictionary_of_lists


class ExperimentEvaluator:

    def __init__(self,
                 model_builders: Dict[str, SegmentationModel]
                 ):
        self.__evaluator = Evaluator(model_builders=model_builders)

    def evaluate_experiment(self, experiment_path: str) -> None:
        to_evaluate = glob(os.path.join(experiment_path, "*", "*"))
        for split_root_path in tqdm(to_evaluate):
            try:
                self.__evaluate_split(split_root_path=split_root_path)
            except Exception as e:
                print(f"For split {split_root_path} there was exception: {e}")

    def __evaluate_split(self, split_root_path: str) -> None:
        model_path = os.path.basename(os.path.dirname(split_root_path))
        metrics, combined_examples = self.__evaluator.evaluate_split(
            model_name=model_path,
            split_root_path=split_root_path
        )
        evaluation_metrics_path = os.path.join(
            split_root_path, EVALUATION_METRICS_FILE_NAME
        )
        dump_json(path=evaluation_metrics_path, content=metrics.to_dict())
        self.__persist_inference_results(
            split_root_path=split_root_path,
            combined_results=combined_examples
        )

    def __persist_inference_results(self,
                                    split_root_path: str,
                                    combined_results: List[Tuple[EvaluationExample, InferenceResult]]
                                    ) -> None:
        target_dir_path = os.path.join(split_root_path, RESULTS_DIR_NAMES)
        os.makedirs(target_dir_path, exist_ok=True)
        for evaluation_example, inference_result in combined_results:
            file_name_root = evaluation_example.example_id
            colored_gt = InferenceWrapper.map_color(
                class_map=np.argmax(evaluation_example.gt, axis=-1)
            )
            cv2.imwrite(
                os.path.join(target_dir_path, f"{file_name_root}_gt.png"),
                colored_gt
            )
            cv2.imwrite(
                os.path.join(target_dir_path, f"{file_name_root}_prediction.png"),
                inference_result.result_colors
            )
            cv2.imwrite(
                os.path.join(target_dir_path, f"{file_name_root}_image.png"),
                ((
                         evaluation_example.image * np.array([0.229, 0.224, 0.225])
                         + np.array([0.485, 0.456, 0.406])) * 255
                 ).astype(np.uint8)[:, :, ::-1]
            )


class Evaluator:

    def __init__(self,
                 model_builders: Dict[str, SegmentationModel]
                 ):
        self.__model_builders = model_builders

    def evaluate_split(self,
                       model_name: str,
                       split_root_path: str
                       ) -> Tuple[EvaluationResults, List[Tuple[EvaluationExample, InferenceResult]]]:
        examples = EvaluationExamplesLoader.load_examples(
            split_root_path=split_root_path
        )
        model = self.__load_model(
            split_root_path=split_root_path,
            model_name=model_name
        )
        results = [
            InferenceWrapper.infer_from_model(model=model, image=e.image)
            for e in examples
        ]
        combined_examples = list(zip(examples, results))
        metrics = MetricsCounter.calculate_metrics(
            calculation_input=combined_examples
        )
        del model
        K.clear_session()
        return metrics, combined_examples

    def __load_model(self,
                     model_name: str,
                     split_root_path: str
                     ) -> Model:
        weights_path = os.path.join(split_root_path, WEIGHTS_FILE_NAME)
        model_builder = self.__model_builders[model_name]
        model = model_builder.build_model(
            input_shape=MODEL_INPUT_SIZE.to_compact_form() + (3, )
        )
        model.load_weights(weights_path)
        return model


class EvaluationExamplesLoader:

    @staticmethod
    def load_examples(split_root_path: str) -> List[EvaluationExample]:
        split = EvaluationExamplesLoader.__load_split_specs(
            split_root_path=split_root_path
        )
        EvaluationExamplesLoader.__assert_split_validity(
            split=split
        )
        return EvaluationExamplesLoader.__load_split_test_examples(
            split=split
        )

    @staticmethod
    def __load_split_specs(split_root_path: str) -> DataSetSplit:
        split_specs_path = os.path.join(
            split_root_path, SPLIT_SPECS_FILE_NAME
        )
        split_specs = parse_json(split_specs_path)
        return DataSetSplit.from_dict(split=split_specs)

    @staticmethod
    def __assert_split_validity(split: DataSetSplit) -> None:
        test_paths = [e.image_path for e in split.test_set.examples]
        train_paths = [e.image_path for e in split.training_set.examples]
        if len(test_paths) != len(set(test_paths)):
            raise ValueError(f"Test examples not unique for {split.name}")
        if len(set(test_paths).intersection(set(train_paths))) > 0:
            raise ValueError(f"Data split missmatch for {split.name}")

    @staticmethod
    def __load_split_test_examples(split: DataSetSplit
                                   ) -> List[EvaluationExample]:
        test_generator = DataGenerator(
            examples=split.test_set.examples,
            transformation_chain=VALIDATION_TRANSFORMATION_CHAIN,
            batch_size=1
        )
        examples = []
        for batch_idx in range(len(test_generator)):
            batch = test_generator[batch_idx]
            example = EvaluationExample(
                image=batch[0][0],
                gt=batch[1][0],
                original_example=split.test_set.examples[batch_idx].to_dict(),
                example_id=f"{uuid4()}"
            )
            examples.append(example)
        return examples


class InferenceWrapper:

    @staticmethod
    def infer_from_model(model: Model, image: np.ndarray) -> InferenceResult:
        image = np.expand_dims(image, axis=0)
        start = time.time()
        result = model.predict_on_batch(image)
        result = np.argmax(np.squeeze(result, axis=0), axis=-1)
        inference_time = time.time() - start
        result_colors = InferenceWrapper.map_color(class_map=result)
        return InferenceResult(
            result=result,
            result_colors=result_colors,
            inference_time=inference_time
        )

    @staticmethod
    def map_color(class_map: np.ndarray) -> np.ndarray:
        return CLASS_TO_COLORS[class_map]


class MetricsCounter:

    @staticmethod
    def calculate_metrics(calculation_input: List[Tuple[EvaluationExample, InferenceResult]]
                          ) -> EvaluationResults:
        metrics_acc = [
            MetricsCounter.calculate_metrics_for_single_example(*e)
            for e in calculation_input
        ]
        metrics_flattened, pixel_acc = MetricsCounter.flatten_metrics(metrics=metrics_acc)
        examples_weighted_class_dice = {
            cls: (float(np.mean(value)), len(value))
            for cls, value in metrics_flattened["dice"].items()
        }
        examples_weighted_class_iou = {
            cls: (float(np.mean(value)), len(value))
            for cls, value in metrics_flattened["iou"].items()
        }
        examples_weighted_class_metrics = {
            "dice": examples_weighted_class_dice,
            "iou": examples_weighted_class_iou
        }
        pixels_weighted_class_dice = {
            cls: (float(np.sum(
                    [e[0] * e[1] for e in zip(
                        metrics_flattened["dice"][cls],
                        metrics_flattened["pixels_voting"][cls]
                        )
                    ]
                ) / np.sum(metrics_flattened["pixels_voting"][cls])),
                int(np.sum(metrics_flattened["pixels_voting"][cls]))
            )
            for cls in metrics_flattened["dice"]
        }
        pixels_weighted_class_iou = {
            cls: (float(np.sum(
                    [e[0] * e[1] for e in zip(
                        metrics_flattened["iou"][cls],
                        metrics_flattened["pixels_voting"][cls]
                    )
                     ]
                ) / np.sum(metrics_flattened["pixels_voting"][cls])),
                int(np.sum(metrics_flattened["pixels_voting"][cls]))
            )
            for cls in metrics_flattened["iou"]
        }
        pixel_weighted_class_metrics = {
            "dice": pixels_weighted_class_dice,
            "iou": pixels_weighted_class_iou
        }
        examples_weighted_mean_dice = float(np.sum([
            mean * weight for mean, weight in examples_weighted_class_dice.values()
        ]) / np.sum([weight for _, weight in examples_weighted_class_dice.values()]))
        examples_weighted_mean_iou = float(np.sum([
            mean * weight for mean, weight in examples_weighted_class_iou.values()
         ]) / np.sum([weight for _, weight in examples_weighted_class_iou.values()]))
        examples_weighted_mean_metrics = {
            "dice": examples_weighted_mean_dice,
            "iou": examples_weighted_mean_iou
        }
        pixel_weighted_mean_dice = float(np.sum([
            mean * weight for mean, weight in pixels_weighted_class_dice.values()
        ]) / np.sum([weight for _, weight in pixels_weighted_class_dice.values()]))
        pixel_weighted_mean_iou = float(np.sum([
            mean * weight for mean, weight in pixels_weighted_class_iou.values()
        ]) / np.sum([weight for _, weight in pixels_weighted_class_iou.values()]))
        pixel_weighted_mean_metrics = {
            "dice": pixel_weighted_mean_dice,
            "iou": pixel_weighted_mean_iou
        }
        mean_pixel_accuracy = float(np.mean(pixel_acc))
        mean_inference_time = float(np.mean([e[1].inference_time for e in calculation_input]))
        inference_time_variance = float(np.var([e[1].inference_time for e in calculation_input]))
        return EvaluationResults(
            results_per_example=metrics_acc,
            examples_weighted_class_metrics=examples_weighted_class_metrics,
            pixel_weighted_class_metrics=pixel_weighted_class_metrics,
            examples_weighted_mean_metrics=examples_weighted_mean_metrics,
            pixel_weighted_mean_metrics=pixel_weighted_mean_metrics,
            mean_pixel_accuracy=mean_pixel_accuracy,
            mean_inference_time=mean_inference_time,
            inference_time_variance=inference_time_variance
        )

    @staticmethod
    def calculate_metrics_for_single_example(evaluation_example: EvaluationExample,
                                             inference_result: InferenceResult,
                                             allowed_prediction_noise: float = 0.0015
                                             ) -> EvaluationResult:
        classes = copy(CLASS_MAPPINGS_REVERTED)
        classes[BACKGROUND_CLASS] = BACKGROUND_CLASS_NAME
        prediction = inference_result.result
        gt = np.argmax(evaluation_example.gt, axis=-1)
        prediction_size = prediction.shape[0] * prediction.shape[1]
        max_noisy_predictions = allowed_prediction_noise * prediction_size
        class_based_metrics = {}
        for c, c_name in classes.items():
            present_in_gt = np.sum(gt == c) > 0
            significant_presence_in_prediction = \
                np.sum(prediction == c) > max_noisy_predictions
            if not present_in_gt and not significant_presence_in_prediction:
                continue
            cm = confusion_matrix(
                (gt == c).flatten(), (prediction == c).flatten(), [0, 1]
            )
            tn, fp, fn, tp = cm.ravel()
            dice = float(2 * tp / (2 * tp + fp + fn) if tp > 0 else 0.0)
            iou = float(tp / (tp + fp + fn) if tp > 0 else 0.0)
            pixels_voting = int(tp + fp + fn)
            class_based_metrics[c_name] = ClassEvaluation(
                dice=dice,
                iou=iou,
                pixels_voting=pixels_voting
            )
        pixel_accuracy = float(np.sum(prediction == gt) / prediction_size)
        return EvaluationResult(
            original_example=evaluation_example.original_example,
            example_id=evaluation_example.example_id,
            class_based_metrics=class_based_metrics,
            pixel_accuracy=pixel_accuracy
        )

    @staticmethod
    def flatten_metrics(metrics: List[EvaluationResult]) -> Tuple[dict, List[float]]:
        per_class_dice = {}
        per_class_pixels = {}
        per_class_iou = {}
        pixel_accs = []
        for result in metrics:
            for cls in result.class_based_metrics:
                per_class_dice = append_to_dictionary_of_lists(
                    dictionary=per_class_dice,
                    to_append=(cls, result.class_based_metrics[cls].dice)
                )
                per_class_pixels = append_to_dictionary_of_lists(
                    dictionary=per_class_pixels,
                    to_append=(cls, result.class_based_metrics[cls].pixels_voting)
                )
                per_class_iou = append_to_dictionary_of_lists(
                    dictionary=per_class_iou,
                    to_append=(cls, result.class_based_metrics[cls].iou)
                )
            pixel_accs.append(result.pixel_accuracy)
        class_flatten_scores = {
            "dice": per_class_dice,
            "iou": per_class_iou,
            "pixels_voting": per_class_pixels,
        }
        return class_flatten_scores, pixel_accs
