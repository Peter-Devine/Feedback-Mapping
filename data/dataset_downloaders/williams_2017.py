import requests
import zipfile
import time
import json
import os
import io
import shutil
import re

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

        def group_follower_count(value):
            # We parse value as int
            value = int(value)

            # Get a grouped value for number of followers, numbers of which are roughly comparable for each bin
            if value < 50:
                return "0-50 followers"
            elif value < 100:
                return "50-100 followers"
            elif value < 200:
                return "100-200 followers"
            elif value < 300:
                return "200-300 followers"
            elif value < 400:
                return "300-400 followers"
            elif value < 500:
                return "400-500 followers"
            else:
                return "500+ followers"

        df["sublabel"] = df["n_followers"].apply(group_follower_count)

        def get_app_name(tweet):
            # Find the name of the software system that the tweet was primarily addressed to. The handle should be the first word in the tweet
            first_word = tweet.split()[0].lower()
            assert first_word[0] == "@", f"First word was expected to be '@' + something but was {first_word} in tweet {tweet} in Williams dataset"

            # We make sure that we only get the handle (it is alphanumeric)
            first_word = "@" + re.split('[^a-zA-Z]', first_word)[1]

            return first_word

        df["app"] = df["text"].apply(get_app_name)

        # Delete all temporary zip files etc.
        shutil.rmtree(task_data_path)

        super(Williams2017, self).download(df)
