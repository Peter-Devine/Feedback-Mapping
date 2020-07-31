import requests
import zipfile
import time
import json
import os
import io
import shutil

import pandas as pd

from data.download_util_base import DownloadUtilBase

class Maalej2016(DownloadUtilBase):
    def download(self):
        task_data_path = os.path.join(self.download_dir, "maalej_2016")

        # Sometimes, retrieving the Maalej dataset results in not getting the zip file, which will be a rate limiting atrifact. So we simply catch any errors and sleep for 10 mins before retrying.
        try:
            # from https://mast.informatik.uni-hamburg.de/wp-content/uploads/2015/06/review_classification_preprint.pdf
            # Bug Report, Feature Request, or Simply Praise? On Automatically Classifying App Reviews
            r = requests.get("https://mast.informatik.uni-hamburg.de/wp-content/uploads/2014/03/REJ_data.zip")
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(path = task_data_path)
        except Exception as err:
            print(f"The following error has been thrown when trying to retreive the Maalej 2016 dataset from mast.informatik.uni-hamburg.de:\n\n{err}\n\nResponse from trying to reach website was:\n{r.content}")
            print(f"Now sleeping for 10 mins and then retrying (N.B. unlimited retries)")
            time.sleep(600)
            return self.download()


        json_path = os.path.join(task_data_path, "REJ_data", "all.json")

        with open(json_path) as json_file:
            data = json.load(json_file)

        shutil.rmtree(task_data_path)

        df_data = [{"title": datum["title"],
                    "comment": datum["comment"],
                    "rating": datum["rating"],
                    "label": datum["label"],
                    "reviewer": datum["reviewer"],
                    "appId": datum["appId"],
                    "dataSource": datum["dataSource"],
                    "date": datum["date"]} for datum in data]

        df = pd.DataFrame(df_data)

        df["text"] = df.comment
        df["sublabel"] = df["rating"]

        # We drop title and comments as they can have bad characters which mess up the csv file
        df = df.drop(["title", "comment"], axis=1)

        # We save the app name in the "app" column to conform to the format of other datasets
        df["app"] = df["appId"]

        super(Maalej2016, self).download(df)
