import torch

from mapping.model_training.transformer_training_nsp import train_nsp
from mapping.model_training.training_data_utils import get_next_sentence_df

from mapping.mapping_models.data_fit_models.masking_lm.bert_masking_trained_mtl import BertMaskingTrainedMtlMapper

from utils.bert_utils import get_lm_embeddings

class BertNspTrainedMtlMapper(BertMaskingTrainedMtlMapper):

    def get_embeds(self):
        test_df = self.get_dataset(dataset_name=self.test_dataset, app_name=self.app_name)

        # We get the embeddings based on the first token position output from the BERT model.
        # This is unlike other embedding methods, wheer wwe take an average.
        # This is because the model is trained to predict next sentence based on the first token position output.
        all_embeddings = get_lm_embeddings(self, test_df, f"{self.get_mapping_name()}", use_first_token_only = True)

        return all_embeddings, test_df

    def set_parameters(self):
        self.model_name = 'bert-base-uncased'
        self.max_length = 256
        self.batch_size = 64
        self.eval_batch_size = 64
        self.lr = 5e-5
        self.eps = 1e-6
        self.wd = 0.01
        self.epochs = 1
        self.patience = 1

    def train_model(self, model_path):
        train_df = self.get_training_data()

        # Get a dataset that contains two pieces of text in every observation. Half of pairs are matched, half are not.
        train_df = get_next_sentence_df(train_df)

        # Save this df for debugging purposes
        self.save_preprocessed_df(train_df, f"{self.test_dataset}_{self.app_name}")

        params = {
            "lr": self.lr,
            "eps": self.eps,
            "wd": self.wd,
            "epochs": self.epochs,
            "patience": self.patience,
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
        }

        model = train_nsp(train_df, params, self.device)

        torch.save(model.state_dict(), model_path)

    def get_mapping_name(self):
        return f"bert_nsp_trained_mtl"
