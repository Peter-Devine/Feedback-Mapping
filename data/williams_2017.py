import requests
import zipfile
import time
import json
import os
import io
import shutil

import pandas as pd

from data.download_util_base import DownloadUtilBase

class Williams2017(DownloadUtilBase):
    def download(self):
        task_data_path = os.path.join(self.download_dir, "williams_2017")
        # from
        # Mining Twitter feeds for software user requirements.
        r = requests.get("http://seel.cse.lsu.edu/data/re17.zip")
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(path = task_data_path)

        file_path = os.path.join(task_data_path, "RE17", "tweets_full_dataset.dat")

        with open(file_path, "r", encoding='ISO-8859-1') as f:
            data = f.read()

        table = pd.read_table(io.StringIO("\n".join(data.split("\n")[16:])), names=["text_data"])

        pos = table["text_data"].apply(lambda x: x.split(",")[0])
        neg = table["text_data"].apply(lambda x: x.split(",")[1])
        feedback_class = table["text_data"].apply(lambda x: x.split(",")[2])
        content = table["text_data"].apply(lambda x: ",".join(x.split(",")[3:-10]).strip("\"") if len(x.split(",")) >= 10 else None)
        feedback_ids = table["text_data"].apply(lambda x: x.split(",")[-10])
        n_favorites = table["text_data"].apply(lambda x: x.split(",")[-9])
        n_followers = table["text_data"].apply(lambda x: x.split(",")[-8])
        n_friends = table["text_data"].apply(lambda x: x.split(",")[-7])
        n_statuses = table["text_data"].apply(lambda x: x.split(",")[-6])
        n_listed = table["text_data"].apply(lambda x: x.split(",")[-5])
        verified = table["text_data"].apply(lambda x: x.split(",")[-4])
        timezone = table["text_data"].apply(lambda x: x.split(",")[-3])
        is_reply = table["text_data"].apply(lambda x: x.split(",")[-2])
        date_posted = table["text_data"].apply(lambda x: x.split(",")[-1])

        df = pd.DataFrame({
            "pos": pos,
            "neg": neg,
            "label": feedback_class,
            "text": content,
            "feedback_ids": feedback_ids,
            "n_favorites": n_favorites,
            "n_followers": n_followers,
            "n_friends": n_friends,
            "n_statuses": n_statuses,
            "n_listed": n_listed,
            "verified": verified,
            "timezone": timezone,
            "is_reply": is_reply,
            "date_posted": date_posted
        })

        shutil.rmtree(task_data_path)

        train_and_val = df.sample(frac=0.7, random_state=self.random_state)
        train = train_and_val.sample(frac=0.7, random_state=self.random_state)
        val = train_and_val.drop(train.index)
        test = df.drop(train_and_val.index)

        super(Williams2017, self).download(train, val, test)
