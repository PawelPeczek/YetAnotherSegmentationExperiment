import os
from google_drive_downloader import GoogleDriveDownloader as gdd
from preprocessing.config import GOOGLE_DRIVE_RESOURCE_ID, RESOURCES_PATH


def fetch():
    gdd.download_file_from_google_drive(file_id=GOOGLE_DRIVE_RESOURCE_ID,
                                        dest_path=str(RESOURCES_PATH) + '.zip',
                                        unzip=True)
