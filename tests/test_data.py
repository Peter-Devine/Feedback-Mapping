import unittest
import os
import shutil

import pandas as pd

from data.guzman_2015 import Guzman2015
from tests.utils import delete_tree

class TestDataDownloaders(unittest.TestCase):

    def test_guzman_2015_download(self):
        data_dir = os.path.join(".", "data", "raw")

        guzman_dir = os.path.join(data_dir, "guzman_2015")

        delete_tree(guzman_dir)

        self.assertFalse(os.path.exists(guzman_dir), msg=f"Failed in deleting all files from {guzman_dir}")

        Guzman2015(random_state=1, download_dir=guzman_dir).download()

        train_dir = os.path.join(guzman_dir, "train.csv")
        val_dir = os.path.join(guzman_dir, "val.csv")
        test_dir = os.path.join(guzman_dir, "test.csv")

        self.assertTrue(os.path.exists(train_dir), msg=f"{train_dir} does not exist")
        self.assertTrue(os.path.exists(val_dir), msg=f"{train_dir} does not exist")
        self.assertTrue(os.path.exists(test_dir), msg=f"{train_dir} does not exist")

        guzman_expected_cols = ["text", "label"]

        self.df_tester(train_dir, expected_columns=guzman_expected_cols)
        self.df_tester(val_dir, expected_columns=guzman_expected_cols)
        self.df_tester(test_dir, expected_columns=guzman_expected_cols)

    def df_tester(self, df_dir, expected_columns):
        try:
            df = pd.read_csv(df_dir)
        except FileNotFoundError as fnferr:
            self.fail(f"Pandas has been unable to read the .csv file at {df_dir} with the following message:\n {fnferr}")
        except Exception as err:
            self.fail(f"Pandas has failed while reading the .csv file at {df_dir} with the following message:\n {err}")

        self.assertEqual(len(df.shape),  2, msg=f"DataFrame extracted from {df_dir} does not have length 2 (length is {len(df.shape)})")
        self.assertTrue(df.shape[0] > 0, msg=f"DataFrame extracted from {df_dir} has less than 0 rows (row number {df.shape[0]})")
        self.assertTrue(df.shape[1] > 0, msg=f"DataFrame extracted from {df_dir} has less than 0 columns (columns number {df.shape[0]})")

        text_dtype = df["text"].dtype
        self.assertEqual(text_dtype, "O", msg=f"Text column is not of string type on {df_dir}. (Text is of dtype {text_dtype})")

        for column in expected_columns:
            self.assertTrue(column in df.columns, msg=f"DataFrame extracted from {df_dir} does not contain the expected column {column} (columns are {df.columns})")

if __name__ == '__main__':
    unittest.main()
