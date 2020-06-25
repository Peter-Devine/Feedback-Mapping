import requests
import zipfile
import time
import json
import os
import io
import sqlite3
import shutil

import pandas as pd
from data.download_util_base import DownloadUtilBase

class Tizard2019(DownloadUtilBase):
    def download(self):

        task_data_path = os.path.join(self.download_dir, "tizard_2019")

        r = requests.get("https://zenodo.org/record/3340156/files/RE_Submission_17-master.zip?download=1")
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(path=task_data_path)

        db = sqlite3.connect(os.path.join(task_data_path, "RE_Submission_17-master", "VLC_labelled_sentences_4RE.sqlite"))

        column_names = [x[1] for x in db.execute("""PRAGMA table_info('labelled_sentences');""")]

        row_data = [row for row in db.execute("SELECT * from labelled_sentences")]

        df = pd.DataFrame(row_data, columns=column_names)

        bad_label_list = [
            "and Remove Cookies Warning\xa0! ",
            "and Remove Cookies  ",
            "remove",
            "continued from previous"
        ]

        # Remove the four rows with mistake labels
        df = df.loc[~df.label.apply(lambda x: x in bad_label_list)]

        def forum_label_transformer(raw_label):
            lower_label = raw_label.lower()

            if "application guidance" in lower_label:
                return "application guidance"
            if "non-informative" in lower_label:
                return "non-informative"
            if "apparent bug" in lower_label:
                return "apparent bug"
            if "question on application" in lower_label:
                return "question on application"
            if "feature request" in lower_label:
                return "feature request"
            if "help seek" in lower_label:
                return "help seeking"
            if "user setup" in lower_label:
                return "user setup"
            if "usage" in lower_label:
                return "application usage"
            if "is-background" in lower_label:
                return "question on background"
            if "attempted solution" in lower_label:
                return "attempted solution"
            if "requesting" in lower_label:
                return "requesting more information"
            if "dispraise" in lower_label:
                return "dispraise for application"
            if "praise application" in lower_label:
                return "praise for application"
            if "acknowledgement" in lower_label:
                return "acknowledgement of problem resolution"
            if "agreeing with the problem" in lower_label:
                return "agreeing with the problem"
            if "limitation confirmation" in lower_label:
                return "limitation confirmation"
            if "bug confirmation" in lower_label:
                return "malfunction confirmation"
            if "agreeing with the request" in lower_label:
                return "agreeing with the feature request"
            return "other"

        # Map all labels to their lower-case proper label. Mark all labels outside of the literature label set as "other".
        df.label = df.label.apply(forum_label_transformer)

        df = df.rename(columns={'sentence': 'text'})

        shutil.rmtree(task_data_path)

        train_and_val = df.sample(frac=0.7, random_state=self.random_state)
        train = train_and_val.sample(frac=0.7, random_state=self.random_state)
        val = train_and_val.drop(train.index)
        test = df.drop(train_and_val.index)

        super(Tizard2019, self).download(train, val, test)
