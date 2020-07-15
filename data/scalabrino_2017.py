import pandas as pd
from data.download_util_base import DownloadUtilBase

class Scalabrino2017(DownloadUtilBase):
    def download(self):

        df = pd.read_csv("https://dibt.unimol.it/reports/clap/downloads/rq3-manually-classified-implemented-reviews.csv")

        df = df.rename(columns = {"body": "text", "category": "label"})
        df["sublabel1"] = df["App-name"]
        df["sublabel2"] = df["rating"]

        # We take out a randomly sampled one of every label to make sure that the training dataset has one label for each class
        unique_df = df.groupby('label',as_index = False,group_keys=False).apply(lambda s: s.sample(1, random_state=self.random_state))
        df = df.drop(unique_df.index)

        train_and_val = df.sample(frac=0.7, random_state=self.random_state)
        train = train_and_val.sample(frac=0.7, random_state=self.random_state)
        val = train_and_val.drop(train.index)
        train = train.append(unique_df)
        test = df.drop(train_and_val.index)

        super(Scalabrino2017, self).download(train, val, test)
