import os
from utils.utils import bad_char_del

class DownloadUtilBase:
    def __init__(self, random_state=None, download_dir="."):
        self.random_state = random_state
        self.download_dir = download_dir
        self.misc_app_name = "MISC_APP"

        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)

    def download(self, df):
        # Each dataset needs to have a "text" column and a "label" column
        # This is so that we know which label contains text...
        # And which column contains a label with which to evaluate
        required_cols = ["text", "label"]
        assert all([required_col in df.columns for required_col in required_cols]), f"Missing one of {required_cols} in df columns ({df.columns})"

        # Clean the text of each dataframe
        df["text"] = df["text"].apply(bad_char_del)

        # If we don't have an app column, then we just mark them all as MISC
        if "app" not in df.columns:
            df["app"] = self.misc_app_name

        # Make sure that apps with less than 100 reviews are labelled as MISC_APP
        # We rename all apps that have less than 100 reviews as MISC_APP as you could quite conceivably just read all 100 reviews instead of relying on ML
        many_review_apps = df.app.value_counts()[df.app.value_counts() >= 100].index
        df["app"] = df["app"].apply(lambda x: x if x in many_review_apps else self.misc_app_name)

        # Iterate over all the unique app names in the dataset, saving an individual csv file of feedback for each app in a given dataset
        for app_name in df.app.unique():
            app_df = df[df.app == app_name]
            app_df.to_csv(os.path.join(self.download_dir, f"{app_name}.csv"))
