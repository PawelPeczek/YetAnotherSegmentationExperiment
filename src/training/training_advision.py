from __future__ import annotations

import os
from typing import List, Tuple, Callable
import logging

from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras import backend as K
import tensorflow as tf

from src.data_access.data_generation import DataGenerator
from src.data_access.data_transformations import DataTransformationChain
from src.data_access.folds_generation import FoldsGenerator
from src.evaluation.losses.dice import ce_dice_loss, dice_loss
from src.models.base import SegmentationModel
from src.primitives.data_access import DataSetSplit
from src.training.persistence_manager import PersistenceManager
from src.utils.iterables import fetch_index_from_list_of_tuples
import src.training.config as training_config


logging.getLogger().setLevel(logging.INFO)

LossFunction = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]


class TrainingAdvisor:

    @classmethod
    def initialize(cls,
                   output_base_dir: str,
                   input_shape: Tuple[int, int, int],
                   batch_size: int,
                   epoch_num: int,
                   optimizer: str,
                   training_set_transformation_chain: DataTransformationChain,
                   test_set_transformation_chain: DataTransformationChain
                   ) -> TrainingAdvisor:
        persistence_manager = PersistenceManager(output_base_dir=output_base_dir)
        return cls(
            persistence_manager=persistence_manager,
            input_shape=input_shape,
            batch_size=batch_size,
            epoch_num=epoch_num,
            optimizer=optimizer,
            training_set_transformation_chain=training_set_transformation_chain,
            test_set_transformation_chain=test_set_transformation_chain,
        )

    def __init__(self,
                 persistence_manager: PersistenceManager,
                 input_shape: Tuple[int, int, int],
                 batch_size: int,
                 epoch_num: int,
                 optimizer: str,
                 training_set_transformation_chain: DataTransformationChain,
                 test_set_transformation_chain: DataTransformationChain,
                 ):
        self.__persistence_manager = persistence_manager
        self.__input_shape = input_shape
        self.__batch_size = batch_size
        self.__epoch_num = epoch_num
        self.__optimizer = optimizer
        self.__training_set_transformation_chain = \
            training_set_transformation_chain
        self.__test_set_transformation_chain = test_set_transformation_chain

    def execute_training(self,
                         experiment_name: str,
                         folds_generator: FoldsGenerator,
                         models_to_train: List[Tuple[str, SegmentationModel, LossFunction]]
                         ) -> None:
        self.__prepare_storage(
            experiment_name=experiment_name,
            models_to_train=models_to_train
        )
        for dataset_split in folds_generator.generate_folds():
            self.__execute_models_training(
                experiment_name=experiment_name,
                models_to_train=models_to_train,
                dataset_split=dataset_split
            )

    def __prepare_storage(self,
                          experiment_name: str,
                          models_to_train: List[Tuple[str, SegmentationModel, LossFunction]]):
        model_names = fetch_index_from_list_of_tuples(
            list_of_tuples=models_to_train,
            index=0
        )
        self.__persistence_manager.prepare_training_storage(
            experiment_name=experiment_name,
            model_names=model_names
        )

    def __execute_models_training(self,
                                  experiment_name: str,
                                  models_to_train: List[Tuple[str, SegmentationModel, LossFunction]],
                                  dataset_split: DataSetSplit
                                  ) -> None:
        for model_name, segmentation_model, loss_function in models_to_train:
            self.__execute_model_training(
                experiment_name=experiment_name,
                model_name=model_name,
                loss_function=loss_function,
                segmentation_model=segmentation_model,
                dataset_split=dataset_split
            )

    def __execute_model_training(self,
                                 experiment_name: str,
                                 model_name: str,
                                 loss_function: LossFunction,
                                 segmentation_model: SegmentationModel,
                                 dataset_split: DataSetSplit
                                 ) -> None:
        logging.info(
            f"Executing training of {model_name} on {dataset_split.name}..."
        )
        segmentation_model = segmentation_model.build_model(
            input_shape=self.__input_shape
        )
        segmentation_model.summary()
        training_set, test_set = \
            dataset_split.training_set, dataset_split.test_set
        logging.info(
            f"Training set class balance: {training_set.classes_balance}"
        )
        logging.info(
            f"Test set class balance: {test_set.classes_balance}"
        )
        logging.info(
            f"Training will be executed with batch size: {self.__batch_size}. "
            f"Number of epochs: {self.__epoch_num}. Optimizer: {self.__optimizer}."
        )
        split_dir = self.__persistence_manager.register_dataset_split_for_model(
            experiment_name=experiment_name,
            model_name=model_name,
            dataset_split=dataset_split
        )
        logging.info(f"Model saving path: {split_dir}")
        checkpoint_callback = ModelCheckpoint(
            os.path.join(split_dir, training_config.WEIGHTS_FILE_NAME),
            save_best_only=True,
            verbose=True
        )
        training_generator = DataGenerator(
            examples=training_set.examples,
            transformation_chain=self.__training_set_transformation_chain,
            batch_size=self.__batch_size
        )
        test_generator = DataGenerator(
            examples=test_set.examples,
            transformation_chain=self.__test_set_transformation_chain,
            batch_size=self.__batch_size
        )
        segmentation_model.compile(
            optimizer="adam",
            loss=loss_function,
            metrics=[dice_loss]
        )
        history = segmentation_model.fit_generator(
            training_generator,
            epochs=self.__epoch_num,
            verbose=1,
            callbacks=[checkpoint_callback],
            validation_data=test_generator,
            max_queue_size=training_config.MAX_QUEUE_SIZE,
            workers=training_config.WORKERS,
            use_multiprocessing=True
        )
        logging.info(f"Training history: {history.history}")
        self.__persistence_manager.persist_model_history(
            split_dir=split_dir, history=history
        )
        del segmentation_model
        K.clear_session()
        logging.info("Model removed from memory.")
        logging.info("Training finished.")
