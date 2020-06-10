import unittest
import os
import shutil

import pandas as pd

from tests.test_data import TestDataDownloaders
from mapping.mapping_models.use import UseMapper
from tests.utils import delete_tree

class TestMappings(unittest.TestCase):

    def test_use_guzman(self):

        TestDataDownloaders().test_guzman_2015_download()

        dataset_name = "guzman_2015"
        mapping_name = "use"

        embed_folder = os.path.join(".", "data", "embeddings", dataset_name)
        embed_path = os.path.join(embed_folder, f"{mapping_name}.csv")

        delete_tree(embed_folder)

        use_mapper = UseMapper(dataset_name)

        use_mapper.embed()

        self.df_tester(embed_path)

    def df_tester(self, df_dir):
        try:
            df = pd.read_csv(df_dir)
        except FileNotFoundError as fnferr:
            self.fail(f"Pandas has been unable to read the .csv file at {df_dir} with the following message:\n {fnferr}")
        except Exception as err:
            self.fail(f"Pandas has failed while reading the .csv file at {df_dir} with the following message:\n {err}")

        self.assertEqual(len(df.shape),  2, msg=f"DataFrame extracted from {df_dir} does not have length 2 (length is {len(df.shape)})")
        self.assertTrue(df.shape[0] > 0, msg=f"DataFrame extracted from {df_dir} has less than 0 rows (row number {df.shape[0]})")
        self.assertTrue(df.shape[1] > 0, msg=f"DataFrame extracted from {df_dir} has less than 0 columns (columns number {df.shape[0]})")

        for column in df.columns:
            if column == "label":
                self.assertEqual(df[column].dtype, "O", msg=f"Dataframe {df_dir}\n Column {column}\n is not of datatype object (actual dtype {df[column].dtype})")
            else:
                self.assertEqual(df[column].dtype, "float64", msg=f"Dataframe {df_dir}\n Column {column}\n is not of datatype float32 (actual dtype {df[column].dtype})")

if __name__ == '__main__':
    unittest.main()
