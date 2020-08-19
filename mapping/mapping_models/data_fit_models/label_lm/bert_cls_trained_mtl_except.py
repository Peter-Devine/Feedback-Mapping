from mapping.mapping_models.data_fit_models.metadata_lm.bert_cls_trained_sublabel_mtl import BertClsTrainedSublabelMtlMapper
from utils.bert_utils import get_lm_embeddings

class BertClsTrainedMtlExceptMapper(BertClsTrainedSublabelMtlMapper):
    # We extend the BertClsTrainedSublabelMtlMapper as it is almost exactly the same as what we want to do here.
    # The key differences are:
    # * We train on "label" instead of "sublabel"
    # * We do not train on any data from the dataset that we are evaluating on
    # * We do not skip any datasets due to them not having "sublabel" in them. (All datasets have "label" columns)

    def get_embeds(self):
        test_df = self.get_dataset(dataset_name=self.test_dataset, app_name=self.app_name)

        all_embeddings = get_lm_embeddings(self, test_df, f"{self.get_mapping_name()}")

        return all_embeddings, test_df

    def set_parameters(self):
        self.model_name = 'bert-base-uncased'
        self.max_length = 128
        self.batch_size = 128
        self.eval_batch_size = 256
        self.lr = 5e-5
        self.eps = 1e-6
        self.wd = 0.01
        self.epochs = 100
        self.patience = 2
        self.training_col = "label"

    def get_model_name(self):
        return f"{self.test_dataset}.pt"

    def get_training_data(self):
        all_datasets_dict = self.get_all_datasets()

        # We remove all the data from the dataset that we are testing, as we cannot test on any in-domain labelled data
        del all_datasets_dict[self.test_dataset]
        return all_datasets_dict

    def get_mapping_name(self):
        return f"bert_cls_trained_mtl_except"
