import math

import pandas as pd

import os

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel
from mapping.mapping_models.mapping_models_base import BaseMapper
from mapping.model_training.transformer_training import train_cls
from mapping.model_training.training_data_utils import get_next_sentence_df
from utils.bert_utils import get_lm_embeddings

class BertNspTrainedMtlMapper(BaseMapper):

    def get_embeds(self):
        test_df = self.get_dataset(self.test_dataset, split="test")

        all_embeddings = get_lm_embeddings(self, test_df, f"{self.get_mapping_name()}")

        return all_embeddings, test_df

    def set_parameters(self):
        self.model_name = 'bert-base-uncased'
        self.max_length = 128
        self.batch_size = 32

    def get_model(self):
        model_path = os.path.join(self.model_dir, f"all.pt")

        model = self.read_or_create_model(model_path)

        return model

    def train_model(self, model_path):
        train_dfs = self.get_all_datasets(split="train")
        valid_dfs = self.get_all_datasets(split="val")

        mtl_datasets = {}
        for dataset_name in train_dfs.keys():
            train_df = get_next_sentence_df(train_dfs[dataset_name])
            valid_df = get_next_sentence_df(valid_dfs[dataset_name])

            mtl_datasets[dataset_name] = (train_df, valid_df)
            self.save_preprocessed_df(train_df, f"{dataset_name}_train")
            self.save_preprocessed_df(valid_df, f"{dataset_name}_val")

        params = {
            "lr": 5e-5,
            "eps": 1e-6,
            "wd": 0.01,
            "epochs": 5,
            "patience": 2
        }

        model = train_cls(mtl_datasets, self.model_name, self.batch_size, self.max_length, self.device, params, training_type="sim_cls")

        torch.save(model.state_dict(), model_path)

    def get_mapping_name(self):
        return f"bert_nsp_trained_mtl"
