import pandas as pd
from download_util_base import DownloadUtilBase

class Guzman2015(DownloadUtilBase):
    def download(self):
        df = pd.read_csv("https://ase.in.tum.de/lehrstuhl_1/images/publications/Emitza_Guzman_Ortega/truthset.tsv",
                         sep="\t", names=[0, "label", 2, "app", 4, "text"])

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

        train_and_val = df.sample(frac=0.7, random_state=self.random_state)
        train = train_and_val.sample(frac=0.7, random_state=self.random_state)
        val = train_and_val.drop(train.index)
        test = df.drop(train_and_val.index)

        super(Guzman2015, self).download(train, val, test)
