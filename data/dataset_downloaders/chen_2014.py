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

        def get_splits_per_app(app_name):
            def df_getter(data_path, label):
                with open(data_path, "r", encoding="iso-8859-1") as f:
                    data = f.read()

                reviews = [" ".join(x.split()[2:]) for x in data.split("\n") if len(x) > 0]
                ratings = [x.split()[1] for x in data.split("\n") if len(x) > 0]

                return pd.DataFrame({"text": reviews,
                                     "label": label,
                                     "app": app_name,
                                     "sublabel": ratings})

            train_info = df_getter(os.path.join(task_data_path, "datasets", app_name, "trainL", "info.txt"), "informative")
            train_noninfo = df_getter(os.path.join(task_data_path, "datasets", app_name, "trainL", "non-info.txt"), "non-informative")

            test_info = df_getter(os.path.join(task_data_path, "datasets", app_name, "test", "info.txt"), "informative")
            test_noninfo = df_getter(os.path.join(task_data_path, "datasets", app_name, "test", "non-info.txt"), "non-informative")

            train_and_val = train_info.append(train_noninfo).reset_index(drop=True)
            test = test_info.append(test_noninfo).reset_index(drop=True)

            return train_and_val, test

        def append_df(original_df, new_df):
            if original_df is None:
                return new_df
            else:
                return original_df.append(new_df).reset_index(drop=True)

        all_train_val = None
        all_test = None
        for app_name in ["swiftkey", "facebook", "tapfish", "templerun2"]:
            app_train_and_val, app_test = get_splits_per_app(app_name)
            all_train_val = append_df(all_train_val, app_train_and_val)
            all_test = append_df(all_test, app_test)

        shutil.rmtree(task_data_path)

        dataset = append_df(all_train_val, all_test)

        super(Chen2014, self).download(dataset)
