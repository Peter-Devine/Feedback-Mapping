from mapping.mapping_models.data_fit_models.metadata_lm.bert_cls_trained_sublabel_dataset import BertClsTrainedSublabelDatasetMapper

class BertClsTrainedSublabelAppMapper(BertClsTrainedSublabelDatasetMapper):
    # Since the app training task is just a subset of the MTL task (but only with one task - with one app in that one task), we simply extend the MTL class, and change the data provided to it.

    def get_model_name(self):
        # Save each datasets model individually
        return f"{self.test_dataset}_{self.app_name}.pt"

    def get_training_data(self):
        # Get the data for one specific app
        df = self.get_dataset(dataset_name=self.test_dataset, app_name=self.app_name)
        return {self.test_dataset: {self.app_name: df}}

    def get_mapping_name(self):
        return f"bert_cls_trained_sublabel_app"
