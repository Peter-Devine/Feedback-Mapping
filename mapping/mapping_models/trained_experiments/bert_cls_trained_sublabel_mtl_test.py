import pandas as pd
import os

from tqdm import tqdm

import torch
from transformers import AutoModel
from mapping.mapping_models.mapping_models_base import BaseMapper
from mapping.model_training.transformer_training import train_cls
from utils.bert_utils import get_lm_embeddings

class BertClsTrainedSublabelMtlTestMapper(BaseMapper):

    def get_embeds(self):
        test_df = self.get_dataset(self.test_dataset, split="test")

        all_embeddings = get_lm_embeddings(self, test_df, f"{self.get_mapping_name()}")

        return all_embeddings, test_df.label

    def set_parameters(self):
        self.model_name = 'bert-base-uncased'
        self.max_length = 128
        self.batch_size = 64

    def get_model(self):
        model_path = os.path.join(self.model_dir, f"all.pt")

        model = self.read_or_create_model(model_path)

        return model

    def train_model(self, model_path):
        train_dfs = self.get_all_datasets(split="train")
        test_dfs = self.get_all_datasets(split="test")
        valid_dfs = self.get_all_datasets(split="val")

        mtl_datasets = {}
        for dataset_name in train_dfs.keys():
            # Get the train and val sets for a given dataset
            train_df, test_df, valid_df = train_dfs[dataset_name], test_dfs[dataset_name], valid_dfs[dataset_name]

            # Add all test data to training data
            train_df = train_df.append(test_df).reset_index(drop=True)

            # Cycle through all sublabels that are given to this dataset
            for sublabel_cols in [x for x in train_df.columns if "sublabel" in x]:

                # For each sublabel, make a new df where the label is that sublabel's values
                sublabel_train_df = train_df.drop("label", axis=1).rename(columns={sublabel_cols: 'label'})
                sublabel_valid_df = valid_df.drop("label", axis=1).rename(columns={sublabel_cols: 'label'})

                # Make sure that valid does not have labels that were not included in train
                sublabel_valid_df = sublabel_valid_df[sublabel_valid_df.label.apply(lambda x: x in sublabel_train_df.label.unique())]

                # Add this to the MTL training dict
                mtl_datasets[f"{dataset_name}_{sublabel_cols}"] = (sublabel_train_df, sublabel_valid_df)

                # Save this df for debugging purposes
                self.save_preprocessed_df(sublabel_train_df, f"{dataset_name}_{sublabel_cols}_train")
                self.save_preprocessed_df(sublabel_valid_df, f"{dataset_name}_{sublabel_cols}_val")

        params = {
            "lr": 5e-5,
            "eps": 1e-6,
            "wd": 0.01,
            "epochs": 5,
            "patience": 2
        }

        model = train_cls(mtl_datasets, self.model_name, self.batch_size, self.max_length, self.device, params)

        torch.save(model.state_dict(), model_path)

    def get_mapping_name(self):
        return f"bert_cls_trained_sublabel_mtl_test"
