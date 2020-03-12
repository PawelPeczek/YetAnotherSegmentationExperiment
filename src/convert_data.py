from src.config import DATASET_PATH
from src.preprocessing.annotations_conversion import AnnotationsConverter

if __name__ == '__main__':
    AnnotationsConverter.convert_all_images(
        dataset_root_path=DATASET_PATH
    )
