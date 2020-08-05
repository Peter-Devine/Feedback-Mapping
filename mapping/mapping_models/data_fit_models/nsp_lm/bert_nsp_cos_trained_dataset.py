import pandas as pd
from mapping.mapping_models.data_fit_models.nsp_lm.bert_nsp_cos_trained_mtl import BertNspCosTrainedMtlMapper
from utils.utils import combine_dict_of_dfs_text

class BertNspCosTrainedDatasetMapper(BertNspCosTrainedMtlMapper):

    def get_model_name(self):
        return f"{self.test_dataset}.pt"

    def get_training_data(self):
        # Create an appended series of text from all dfs from all datasets and apps

        # Get the data of all apps in this dataset
        dataset_dict = self.get_all_app_data(self.test_dataset)

        dataset_text_series = combine_dict_of_dfs_text(dataset_dict)

        return pd.DataFrame({"text": dataset_text_series})

    def get_mapping_name(self):
        return f"bert_nsp_cos_trained_dataset"
