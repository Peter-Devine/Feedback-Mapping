import pandas as pd
import os

from tqdm import tqdm

import torch
from transformers import AutoModel
from mapping.mapping_models.mapping_models_base import BaseMapper
from mapping.model_training.transformer_training import train_cls
from utils.bert_utils import get_lm_embeddings

class BertClsTrainedSublabelTestMapper(BaseMapper):

    def get_embeds(self):
        test_df = self.get_dataset(self.test_dataset, split="test")

        all_embeddings = get_lm_embeddings(self, test_df, f"{self.get_mapping_name()}")

        return all_embeddings, test_df

    def set_parameters(self):
        self.model_name = 'bert-base-uncased'
        self.max_length = 128
        self.batch_size = 64

    def get_model(self):
        model_path = os.path.join(self.model_dir, f"{self.test_dataset}.pt")

        model = self.read_or_create_model(model_path)

        return model

    def train_model(self, model_path):
        train_df = self.get_dataset(self.test_dataset, split="train")
        test_df = self.get_dataset(self.test_dataset, split="test")
        train_df = train_df.append(test_df).reset_index(drop=True)
        valid_df = self.get_dataset(self.test_dataset, split="val")

        assert "sublabel1" in train_df.columns, f"'sublabel1' is not in columns {train_df.columns} in dataset {self.test_dataset}. Cannot perform sublabel classification."
        train_df["label"] = train_df["sublabel1"]
        valid_df["label"] = valid_df["sublabel1"]

        # Make sure that valid does not have labels that were not included in train
        valid_df = valid_df[valid_df.label.apply(lambda x: x in train_df.label.unique())]

        self.save_preprocessed_df(train_df, f"{self.test_dataset}_train")
        self.save_preprocessed_df(valid_df, f"{self.test_dataset}_val")

        training_data_dict = {self.test_dataset: (train_df, valid_df)}

        params = {
            "lr": 5e-5,
            "eps": 1e-6,
            "wd": 0.01,
            "epochs": 2+int(10000/train_df.shape[0]),
            "patience": 2
        }

        model = train_cls(training_data_dict, self.model_name, self.batch_size, self.max_length, self.device, params)

        torch.save(model.state_dict(), model_path)

    def get_mapping_name(self):
        return f"bert_cls_trained_sublabel_test"
