import os
from data.guzman_2015 import Guzman2015
from data.maalej_2016 import Maalej2016
from data.williams_2017 import Williams2017
from data.chen_2014 import Chen2014
from data.di_sorbo_2016 import DiSorbo2016
from data.scalabrino_2017 import Scalabrino2017
from data.jha_2017 import Jha2017
from data.tizard_2019 import Tizard2019
from data.morales_ramirez_2019 import MoralesRamirez2019

from utils.utils import get_random_seed

def get_data(list_of_data):
    for dataset in list_of_data:
        get_single_dataset(dataset)

def get_single_dataset(dataset):

    DOWNLOADER_DICT = {
        "guzman_2015": Guzman2015,
        "maalej_2016": Maalej2016,
        "williams_2017": Williams2017,
        "chen_2014": Chen2014,
        "di_sorbo_2016": DiSorbo2016,
        "scalabrino_2017": Scalabrino2017,
        "jha_2017": Jha2017,
        "tizard_2019": Tizard2019,
        "morales_ramirez_2019": MoralesRamirez2019,
    }

    DOWNLOAD_DIR = os.path.join(".", "data", "raw")
    if not os.path.exists(DOWNLOAD_DIR):
        os.mkdir(DOWNLOAD_DIR)

    dataset_download_dir = os.path.join(DOWNLOAD_DIR, dataset)
    if not os.path.exists(dataset_download_dir):
        assert dataset in DOWNLOADER_DICT.keys(), f"{dataset} is not supported. Please create a folder with the path {dataset_download_dir} and add your own train.csv, val.csv and test.csv files with 'text' and 'label' columns to use your own data."
        downloader = DOWNLOADER_DICT[dataset](random_state = get_random_seed(), download_dir = dataset_download_dir)
        print(f"Downloading {dataset}")
        downloader.download()
