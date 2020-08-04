import pandas as pd
from mapping.mapping_models.data_fit_models.nsp_lm.bert_nsp_cos_trained_mtl import BertNspCosTrainedMtlMapper

class BertNspCosTrainedDatasetMapper(BertNspCosTrainedMtlMapper):

    def get_model_name(self):
        return f"{self.test_dataset}.pt"

    def get_training_data(self):
        # Create an appended series of text from all dfs from all datasets and apps
        dataset_series = None

        # Get the data of all apps in this dataset
        dataset_dict = self.get_all_app_data(self.test_dataset)

        # Iterate over each app in dataset
        for app, app_df in dataset_dict.items():

            if dataset_series is None:
                dataset_series = app_df.text
            else:
                dataset_series = dataset_series.append(app_df.text).reset_index(drop=True)

        return pd.DataFrame({"text": dataset_series})

    def get_mapping_name(self):
        return f"bert_nsp_cos_trained_dataset"
