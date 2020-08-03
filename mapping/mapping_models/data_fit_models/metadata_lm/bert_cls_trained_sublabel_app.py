from mapping.mapping_models.data_fit_models.metadata_lm.bert_cls_trained_sublabel_mtl import BertClsTrainedSublabelMtlMapper

class BertClsTrainedSublabelAppMapper(BertClsTrainedSublabelMtlMapper):
    # Since the app training task is just a subset of the MTL task (but only with one task - with one app in that one task), we simply extend the MTL class, and change the data provided to it.

    def get_embeds(self):
        # Only run embedding if app is not MISC_APP. MISC_APP is a mixture of lots of feedback from many reviews, but not enough to train an ML model for any individual one. Evaluating over MISC_APP is meaningless.
        if self.app_name == self.misc_app_name:
            return None
        else:
            return super().get_embeds()

    def get_model_name(self):
        # Save each datasets model individually
        return f"{self.test_dataset}_{self.app_name}.pt"

    def get_training_data(self):
        # Get the data for one specific app
        df = self.get_dataset(dataset_name=self.test_dataset, app_name=self.app_name)
        return {self.test_dataset: {self.app_name: df}}

    def get_mapping_name(self):
        return f"bert_cls_trained_sublabel_app"
