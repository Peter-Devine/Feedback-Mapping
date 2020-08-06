from mapping.mapping_models.data_fit_models.nsp_lm.bert_nsp_trained_mtl import BertNspTrainedMtlMapper

class BertNspTrainedAppMapper(BertNspTrainedMtlMapper):

    def get_model_name(self):
        return f"{self.test_dataset}_{self.app_name}.pt"

    def get_training_data(self):
        # Just get the data needed for one app
        df = self.get_dataset(dataset_name=self.test_dataset, app_name=self.app_name)

        return pd.DataFrame({"text": df.text})

    def get_mapping_name(self):
        return f"bert_nsp_trained_app"
