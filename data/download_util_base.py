import os

class DownloadUtilBase:
    def __init__(self, random_state, download_dir):
        self.random_state = random_state
        self.download_dir = download_dir

    def download(self, train, val, test):
        train.to_csv(os.path.join(self.download_dir, "train.csv"))
        val.to_csv(os.path.join(self.download_dir, "val.csv"))
        test.to_csv(os.path.join(self.download_dir, "test.csv"))
