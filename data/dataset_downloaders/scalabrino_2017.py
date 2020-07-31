import pandas as pd
from data.download_util_base import DownloadUtilBase

class Scalabrino2017(DownloadUtilBase):
    def download(self):

        df = pd.read_csv("https://dibt.unimol.it/reports/clap/downloads/rq3-manually-classified-implemented-reviews.csv")

        df = df.rename(columns = {"body": "text", "category": "label"})
        df["app"] = df["App-name"]
        df["sublabel"] = df["rating"]

        super(Scalabrino2017, self).download(df)
