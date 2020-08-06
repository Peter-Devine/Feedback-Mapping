import os
import torch

from mapping.mapping_models.mapping_models_base import BaseMapper
from mapping.model_training.transformer_training_cls import train_cls
from utils.bert_utils import get_lm_embeddings

class BertClsTrainedSublabelMtlMapper(BaseMapper):

    def get_embeds(self):
        test_df = self.get_dataset(dataset_name=self.test_dataset, app_name=self.app_name)

        all_embeddings = get_lm_embeddings(self, test_df, f"{self.get_mapping_name()}")

        return all_embeddings, test_df

    def set_parameters(self):
        self.model_name = 'bert-base-uncased'
        self.max_length = 128
        self.batch_size = 64
        self.eval_batch_size = 128
        self.lr = 5e-5
        self.eps = 1e-6
        self.wd = 0.01
        self.epochs = 30
        self.patience = 1
        self.training_col = "sublabel"

    def get_model(self):
        model_path = os.path.join(self.model_dir, self.get_model_name())

        model = self.read_or_create_model(model_path)

        return model

    def get_model_name(self):
        return "all.pt"

    def get_training_data(self):
        return self.get_all_datasets()

    def train_model(self, model_path):
        train_dfs = self.get_training_data()

        mtl_datasets = self.prepare_train_tasks_dataset(train_dfs)

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

        model = train_cls(mtl_datasets, params, self.device)

        torch.save(model.state_dict(), model_path)

    def prepare_train_tasks_dataset(self, train_dfs_dict):
        mtl_datasets = {}

        for dataset_name in train_dfs_dict.keys():
            # Get the dfs for a given dataset
            app_train_dfs_dict = train_dfs_dict[dataset_name]

            # Combine all app data for one dataset into a single df
            train_df = None
            dataset_contains_training_col = True
            for app_name, app_df in app_train_dfs_dict.items():

                # We skip any datasets that do not have the columns that we are looking to train on
                if self.training_col not in app_df.columns:
                    dataset_contains_training_col = False
                    break

                if train_df is None:
                    train_df = app_df
                else:
                    train_df = train_df.append(app_df).reset_index(drop=True)

            # If the dfs for this dataset do not contain the column needed to train a classification model, then we do not add this dataset to our training set
            if not dataset_contains_training_col:
                continue

            # Train the classifier model to predict sublabel
            train_df["label"] = train_df[self.training_col]

            mtl_datasets[dataset_name] = train_df

            # Save this df for debugging purposes
            self.save_preprocessed_df(train_df, f"{dataset_name}_{self.app_name}")

        return mtl_datasets

    def get_mapping_name(self):
        return f"bert_cls_trained_sublabel_mtl"
