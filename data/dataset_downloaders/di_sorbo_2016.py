import requests
import zipfile
import time
import json
import os
import io
import shutil

from bs4 import BeautifulSoup

import pandas as pd
from data.download_util_base import DownloadUtilBase

class DiSorbo2016(DownloadUtilBase):
    def download(self):

        task_data_path = os.path.join(self.download_dir, "di_sorbo_2017")
        # from https://www.merlin.uzh.ch/contributionDocument/download/9373
        # What Would Users Change in My App? Summarizing App Reviews for Recommending Software Changes
        r = requests.get("https://zenodo.org/record/47323/files/SURF-SURF-v.1.0.zip?download=1")
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(path=task_data_path)

        file_dir = os.path.join(task_data_path, "panichella-SURF-29332ec", "SURF_replication_package", "Experiment I", "summaries")

        def add_data_to_df(data, df, app_name):
            soup = BeautifulSoup(data, 'html.parser')

            label_upper_element = [x for x in soup.find_all("sup")]
            text_list = [x.findNext('a').text for x in label_upper_element]
            aspect_list = [x.findPrevious('h2').text for x in label_upper_element]
            label_list = [x.find('b').text for x in label_upper_element]

            full_review_df = pd.DataFrame({"text": text_list, "label": label_list, "app": app_name})

            if df is None:
                df = full_review_df
            else:
                df = df.append(full_review_df).reset_index(drop=True)

            return df

        # Cycle through all the app files and add the review text and labels to the overall dataframe
        all_review_df = None
        for file_name in os.listdir(file_dir):
            with open(os.path.join(file_dir, file_name), "rb") as f:
                data = f.read()
                all_review_df = add_data_to_df(data, all_review_df, file_name)

        shutil.rmtree(task_data_path)

        super(DiSorbo2016, self).download(all_review_df)
