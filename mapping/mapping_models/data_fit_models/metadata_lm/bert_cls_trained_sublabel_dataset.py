from mapping.mapping_models.data_fit_models.metadata_lm.bert_cls_trained_sublabel_mtl import BertClsTrainedSublabelMtlMapper

class BertClsTrainedSublabelDatasetMapper(BertClsTrainedSublabelMtlMapper):
    # Since the dataset training task is just a subset of the MTL task (but only with one task), we simply extend the MTL class, and change the data provided to it.

    def get_model_name(self):
        # Save each datasets model individually
        return f"{self.test_dataset}.pt"

    def get_training_data(self):
        # Get all app data for a specific dataset
        return {self.test_dataset: self.get_all_app_data(dataset_name=self.test_dataset)}

    def get_mapping_name(self):
        return f"bert_cls_trained_sublabel_dataset"
