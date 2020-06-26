import pandas as pd
import requests
import zipfile
import io
import os
import shutil
from data.download_util_base import DownloadUtilBase

class MoralesRamirez2019(DownloadUtilBase):
    def download(self):

        task_data_path = os.path.join(self.download_dir, "morales_ramirez_2019")

        def get_confirm_token(response):
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    return value

            return None

        URL = "https://docs.google.com/uc?export=download"

        id_ = "1PIEAY3o1RGNiIVeASYSoKtTqRXPJxsQ6"

        session = requests.Session()

        response = session.get(URL, params = { 'id' : id_ }, stream = True)
        token = get_confirm_token(response)

        if token:
            params = { 'id' : id_, 'confirm' : token }
            r = session.get(URL, params = params, stream = True)

        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(path=task_data_path)

        csv_file_location = os.path.join(task_data_path, "FilteredComments3Classes01Nov2017.csv")

        df = pd.read_csv(csv_file_location, names=["id", "unknown", "text", "label", "is_defect"], encoding='ISO-8859-1')

        # Removing lines of code, log files and urls from dataset - we can detect these with easier methods, so we just want to focus on natural language
        df = df[~df.label.apply(lambda x: any([bad_label in x for bad_label in ["CODE_LINE", "LOG_FILE", "URL_link"]]))]

        # Currently, there are > 100 unique labels, but they are just joins of each other (E.g. thanks_AND_informative) (only for a small set of data, most observations are single label)
        # So we squash it down so that any text with multiple labels now is listed as an observation multiple times, with only one label at any one time.
        df = repeat_multi_label_rows(df, "label", "_AND_")

        shutil.rmtree(task_data_path)

        train_and_val = df.sample(frac=0.7, random_state=self.random_state)
        train = train_and_val.sample(frac=0.7, random_state=self.random_state)
        val = train_and_val.drop(train.index)
        test = df.drop(train_and_val.index)

        super(MoralesRamirez2019, self).download(train, val, test)

def repeat_multi_label_rows(input_df, multi_label_row_column, split_str):

    df = input_df

    df[multi_label_row_column] = df[multi_label_row_column].str.split(split_str)

    non_ml_cols = [column for column in df.columns if column != multi_label_row_column]

    return (df
     .set_index(non_ml_cols)[multi_label_row_column]
     .apply(pd.Series)
     .stack()
     .reset_index()
     .drop(f'level_{len(non_ml_cols)}', axis=1)
     .rename(columns={0: multi_label_row_column}))
