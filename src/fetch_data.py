import os

from google_drive_downloader import GoogleDriveDownloader
from src.config import RESOURCES_PATH, GOOGLE_DRIVE_RESOURCE_ID, DATASET_PATH


def fetch():
    target_path = os.path.join(RESOURCES_PATH, 'data_set.zip')
    if os.path.exists(target_path) or os.path.exists(DATASET_PATH):
        raise RuntimeError("Data set already fetched.")
    GoogleDriveDownloader.download_file_from_google_drive(
        file_id=GOOGLE_DRIVE_RESOURCE_ID,
        dest_path=target_path,
        unzip=True
    )


if __name__ == '__main__':
    fetch()
