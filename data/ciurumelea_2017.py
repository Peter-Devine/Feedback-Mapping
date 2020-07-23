import requests
import zipfile
import os
import io
import shutil

import pandas as pd
import numpy as np
from data.download_util_base import DownloadUtilBase

class Ciurumelea2017(DownloadUtilBase):
    def download(self):

        task_data_path = os.path.join(self.download_dir, "ciurumelea_2017")
        # from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7884612&tag=1
        # Analyzing Reviews and Code of Mobile Apps for Better Release Planning
        r = requests.get("https://zenodo.org/record/161842/files/panichella/UserReviewReference-Replication-Package-URR-v1.0.zip?download=1")
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(path=task_data_path)

        review_data_path = os.path.join(task_data_path, "panichella-UserReviewReference-Replication-Package-643afe0", "data", "reviews", "golden_set.csv")

        df = pd.read_csv(review_data_path, encoding="iso-8859-1")

        shutil.rmtree(task_data_path)

        df["text"] = df.reviewText
        df["sublabel1"] = df.ratingStars
        df["sublabel2"] = df.app

        # Make the multi-label dataset a repeated single-label dataset
        # E.g. one observation with the labels "SECURITY" and "BATTERY" would become two identical rows, but one with the label "SECURITY" and the other the label "BATTERY"
        lst_col = "fine_label"

        df[lst_col] = df.apply(lambda x: list(x[["class1", "class2", "class3", "class4", "class5"]]), axis=1)

        repeated_df = pd.DataFrame({
              col:np.repeat(df[col].values, df[lst_col].str.len())
              for col in df.columns.drop(lst_col)}
            ).assign(**{lst_col:np.concatenate(df[lst_col].values)})[df.columns]

        repeated_df = repeated_df[repeated_df[lst_col] != 'nan']

        # This taxonomy is taken from the paper
        coarse_to_fine_dict = {
            "COMPATIBILITY": ['DEVICE', 'ANDROID VERSION', 'HARDWARE'],
            "USAGE": ['APP USABILITY', 'UI'],
            "RESSOURCES": ['PERFORMANCE', 'BATTERY', 'MEMORY'],
            "PRICING": ['LICENSING', 'PRICE'],
            "PROTECTION": ['SECURITY', 'PRIVACY'],
            "OTHER": ['ERROR', "OTHER"]
        }

        fine_to_coarse_dict = {value: key for key in coarse_to_fine_dict for value in coarse_to_fine_dict[key]}

        repeated_df["label"] = repeated_df[lst_col].apply(lambda x: fine_to_coarse_dict[x])

        train_and_val = repeated_df.sample(frac=0.7, random_state=self.random_state)
        train = train_and_val.sample(frac=0.7, random_state=self.random_state)
        val = train_and_val.drop(train.index)
        test = repeated_df.drop(train_and_val.index)

        super(Ciurumelea2017, self).download(train, val, test)
