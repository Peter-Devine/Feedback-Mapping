import requests
import zipfile
import time
import json
import os
import io
import shutil

import pandas as pd
from data.download_util_base import DownloadUtilBase

class Chen2014(DownloadUtilBase):
    def download(self):

        task_data_path = os.path.join(self.download_dir, "chen_2014")
        # from https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=3323&context=sis_research
        # AR-Miner: Mining Informative Reviews for Developers from Mobile App Marketplace
        r = requests.get("https://sites.google.com/site/appsuserreviews/home/datasets.zip?attredirects=0&d=1")
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(path=task_data_path)

        def df_getter(data_path, label):
            with open(data_path, "r") as f:
                data = f.read()
            return pd.DataFrame({"text": [" ".join(x.split()[2:]) for x in data.split("\n") if len(x) > 0],
                          "label": label})

        train_info = df_getter(os.path.join(task_data_path, "datasets", "swiftkey", "trainL", "info.txt"), "informative")
        train_noninfo = df_getter(os.path.join(task_data_path, "datasets", "swiftkey", "trainL", "non-info.txt"), "non-informative")

        test_info = df_getter(os.path.join(task_data_path, "datasets", "swiftkey", "test", "info.txt"), "informative")
        test_noninfo = df_getter(os.path.join(task_data_path, "datasets", "swiftkey", "test", "non-info.txt"), "non-informative")

        shutil.rmtree(task_data_path)

        train_and_val = train_info.append(train_noninfo)

        train = train_and_val.sample(frac=0.7, random_state=self.random_state)
        val = train_and_val.drop(train.index)
        test = test_info.append(test_noninfo)

        super(Chen2014, self).download(train, val, test)
