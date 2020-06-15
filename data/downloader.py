import os
from data.guzman_2015 import Guzman2015
from data.maalej_2016 import Maalej2016
from data.williams_2017 import Williams2017

def get_data(list_of_data, random_state):

    DOWNLOADER_DICT = {
        "guzman_2015": Guzman2015,
        "maalej_2016": Maalej2016,
        "williams_2017": Williams2017,
    }

    DOWNLOAD_DIR = os.path.join(".", "data", "raw")

    for dataset in list_of_data:
        dataset_download_dir = os.path.join(DOWNLOAD_DIR, dataset)
        if not os.path.exists(dataset_download_dir):
            downloader = DOWNLOADER_DICT[dataset](random_state = random_state, download_dir = dataset_download_dir)
            downloader.download()
