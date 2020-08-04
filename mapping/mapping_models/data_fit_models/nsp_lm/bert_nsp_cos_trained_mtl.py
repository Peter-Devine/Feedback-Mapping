import os
import pandas as pd

from mapping.mapping_models.mapping_models_base import BaseMapper
from mapping.model_training.training_data_utils import get_next_sentence_df
from mapping.model_training.transformer_training_nsp_cos import train_nsp_cos
from utils.bert_utils import get_lm_embeddings

class BertNspCosTrainMtlMapper(BaseMapper):

    def get_embeds(self):
        test_df = self.get_dataset(dataset_name=self.test_dataset, app_name=self.app_name)

        all_embeddings = get_lm_embeddings(self, test_df, f"{self.get_mapping_name()}")

        return all_embeddings, test_df

    def set_parameters(self):
        self.model_name = 'bert-base-uncased'
        self.max_length = 128
        self.batch_size = 32
        self.lr = 5e-5
        self.eps = 1e-6
        self.wd = 0.01
        self.epochs = 30
        self.patience = 1

    def get_model(self):
        model_path = os.path.join(self.model_dir, self.get_model_name())

        model = self.read_or_create_model(model_path)

        return model

    def get_model_name(self):
        return "all.pt"

    def get_training_data(self):
        # Create an appended series of text from all dfs from all datasets and apps
        all_dataset_series = None

        all_dataset_dict = self.get_all_datasets()
        # Iterate over each dataset
        for dataset, dataset_dict in all_dataset_dict.items():
            # Iterate over each app in dataset
            for app, app_df in dataset_dict.items():

                # Append data to all_dataset_series
                if all_dataset_series is None:
                    all_dataset_series = app_df.text
                else:
                    all_dataset_series = all_dataset_series.append(app_df.text).reset_index(drop=True)

        return pd.DataFrame({"text": all_dataset_series})

    def train_model(self, model_path):
        train_df = self.get_training_data()

        mtl_format_dataset = self.prepare_train_tasks_dataset(train_df)

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

        model = train_nsp_cos(mtl_format_dataset, params, self.device)

        torch.save(model.state_dict(), model_path)

    def prepare_train_tasks_dataset(self, all_train_df):
        # Get a dataset that contains two pieces of text in every observation. Half of pairs are matched, half are not.
        nsp_train_df = get_next_sentence_df(all_train_df)

        # Save this df for debugging purposes
        self.save_preprocessed_df(nsp_train_df, f"{self.test_dataset}_{self.app_name}")

        # Get this df into a format that the training expects it to be in (a dict with the key as the task name and the value as the training data)
        mtl_format_dataset = {"all_datasets": nsp_train_df}

        return mtl_format_dataset

    def get_mapping_name(self):
        return f"bert_nsp_cos_trained_mtl"
