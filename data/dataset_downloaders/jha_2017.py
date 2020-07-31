import requests
import zipfile
import time
import json
import os
import io
import shutil

import pandas as pd
from data.download_util_base import DownloadUtilBase

class Jha2017(DownloadUtilBase):
    def download(self):

        task_data_path = os.path.join(self.download_dir, "jha_2017")
        # from https://www.springer.com/content/pdf/10.1007%2F978-3-319-54045-0.pdf
        # Mining User Requirements from Application Store Reviews Using Frame Semantics
        r = requests.get("http://seel.cse.lsu.edu/data/refsq17.zip")
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(path=task_data_path)

        def get_jha_df(filename):
            review_data_path = os.path.join(task_data_path, "refsq17", "refsq17", "BOW", filename)

            with open(review_data_path, "r") as f:
                text_data = f.read()

            df = pd.read_csv(io.StringIO(text_data), names=["text", "label"])

            # Strip out unnecessary ' that bookends every review
            df.text = df.text.apply(lambda x: x.strip("'"))
            # Strip the unnecessary whitespace that prepends every label
            df.label = df.label.apply(lambda x: x.strip())
            return df

        train_and_val = get_jha_df("BOW_training.txt")
        test = get_jha_df("BOW_testing.txt")
        full_df = train_and_val.append(test).reset_index(drop=True)

        shutil.rmtree(task_data_path)

        super(Jha2017, self).download(full_df)
