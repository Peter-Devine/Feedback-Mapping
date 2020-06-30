import os

class DownloadUtilBase:
    def __init__(self, random_state, download_dir):
        self.random_state = random_state
        self.download_dir = download_dir

        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)

    def download(self, train, val, test):
        bad_char_del = lambda x: x.replace("\n", " ").replace("\r", " ").replace("\t", " ")

        train["text"] = train["text"].apply(bad_char_del)
        val["text"] = val["text"].apply(bad_char_del)
        test["text"] = test["text"].apply(bad_char_del)

        train.to_csv(os.path.join(self.download_dir, "train.csv"))
        val.to_csv(os.path.join(self.download_dir, "val.csv"))
        test.to_csv(os.path.join(self.download_dir, "test.csv"))
