import pandas as pd
from data.download_util_base import DownloadUtilBase

class Guzman2015(DownloadUtilBase):
    def download(self):
        df = pd.read_csv("https://ase.in.tum.de/lehrstuhl_1/images/publications/Emitza_Guzman_Ortega/truthset.tsv",
                         sep="\t", names=[0, "label", 2, "app", "rating", "text"])

        int_to_str_label_map = {
            5: "Praise",
            3: "Feature shortcoming",
            1: "Bug report",
            2: "Feature strength",
            7: "Usage scenario",
            4: "User request",
            6: "Complaint",
            8: "Noise"
        }
        df["label"] = df.label.apply(lambda x: int_to_str_label_map[x])
        df["sublabel"] = df["rating"]

        int_to_app_name_map = {
            6: "Picsart",
            8: "Whatsapp",
            7: "Pininterest",
            1: "Angrybirds",
            3: "Evernote",
            5: "Tripadvisor",
            2: "Dropbox"
        }

        df["app"] = df.app.apply(lambda x: int_to_app_name_map[x])

        super(Guzman2015, self).download(df)
