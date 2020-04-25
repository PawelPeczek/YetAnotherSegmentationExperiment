import os
from functools import partial
from typing import List

from tensorflow.python.keras.callbacks import History

from src.primitives.data_access import DataSetSplit
from src.utils.fs_utils import prepare_storage_dir, escape_base_name, dump_json
from src.utils.iterables import for_each
import src.training.config as training_config


class PersistenceManager:

    def __init__(self, output_base_dir: str):
        self.__output_base_dir = output_base_dir

    def prepare_training_storage(self,
                                 experiment_name: str,
                                 model_names: List[str]
                                 ) -> None:
        experiment_name = escape_base_name(base_name=experiment_name)
        experiment_dir = os.path.join(self.__output_base_dir, experiment_name)
        if os.path.exists(experiment_dir):
            raise RuntimeError(
                f"Experiment with name {experiment_name} was already executed. "
                f"Use unique experiment names to avoid overriding or clean the "
                f"storage under path: {experiment_dir}"
            )
        prepare_storage_dir(path=experiment_dir)
        prepare_model_storage = partial(
            self.__prepare_model_storage, experiment_dir=experiment_dir
        )
        for_each(iterable=model_names, side_effect=prepare_model_storage)

    def register_dataset_split_for_model(self,
                                         experiment_name: str,
                                         model_name: str,
                                         dataset_split: DataSetSplit
                                         ) -> str:
        model_name = escape_base_name(base_name=model_name)
        split_name = escape_base_name(base_name=dataset_split.name)
        split_dir = os.path.join(
            self.__output_base_dir, experiment_name, model_name, split_name
        )
        if os.path.exists(split_dir):
            raise RuntimeError(
                f"Split with name {split_name} was already reported for model "
                f"{model_name}. Use unique split names within experiment "
                f"to avoid results overriding."
            )
        prepare_storage_dir(path=split_dir)
        split_specs_path = os.path.join(
            split_dir, training_config.SPLIT_SPECS_FILE_NAME
        )
        dump_json(path=split_specs_path, content=dataset_split.to_dict())
        return split_dir

    def persist_model_history(self, split_dir: str, history: History) -> None:
        history_path = os.path.join(split_dir, training_config.HISTORY_FILE_NAME)
        history_dict = {
            key: [v.item() for v in value]
            for key, value in history.history.items()
        }
        dump_json(path=history_path, content=history_dict)

    def __prepare_model_storage(self,
                                model_name: str,
                                experiment_dir: str
                                ) -> None:
        model_name = escape_base_name(base_name=model_name)
        model_training_path = os.path.join(experiment_dir, model_name)
        if os.path.exists(model_training_path):
            raise RuntimeError(
                f"Model with name {model_name} was already reported within "
                f"{experiment_dir}. Use unique model names within experiment "
                f"to avoid results overriding."
            )
        prepare_storage_dir(path=model_training_path)
