import os
from data.guzman_2015 import Guzman2015

def get_data(list_of_data, random_state):

    downloaders = {
        "guzman_2015": Guzman2015
    }

    DOWNLOAD_DIR = os.path.join(".", "data", "raw")

    for dataset in list_of_data:
        dataset_download_dir = os.path.join(DOWNLOAD_DIR, dataset)
        if not os.path.exists(dataset_download_dir):
            downloader = downloaders[dataset](random_state = random_state, download_dir = dataset_download_dir)
            downloader.download()
