import math

import pandas as pd

import os

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel
from mapping.mapping_models.mapping_models_base import BaseMapper
from mapping.model_training.transformer_training import train_sim
from mapping.model_training.training_data_utils import get_next_sentence_df
from utils.bert_utils import get_lm_embeddings

class BertNspTrainedMtlMapper(BaseMapper):

    def get_embeds(self):
        test_df = self.get_dataset(self.test_dataset, split="test")

        all_embeddings = get_lm_embeddings(self, test_df, f"{self.get_mapping_name()}")

        return all_embeddings, test_df.label

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

        def get_concat_next_sentence_df(dfs):
            concat_df = None

            for dataset_name, dataset_df in dfs.items():
                two_sent_df = get_next_sentence_df(dataset_df)

                if concat_df is None:
                    concat_df = two_sent_df
                else:
                    concat_df = concat_df.append(two_sent_df)

            return concat_df

        train_df = get_concat_next_sentence_df(train_dfs)
        valid_df = get_concat_next_sentence_df(valid_dfs)

        self.save_preprocessed_df(train_df, f"train")
        self.save_preprocessed_df(valid_df, f"val")

        params = {
            "lr": 5e-5,
            "eps": 1e-6,
            "wd": 0.01,
            "epochs": 5,
            "patience": 2
        }

        model = train_sim(train_df, valid_df, self.model_name, self.batch_size, self.max_length, self.device, params)

        torch.save(model.state_dict(), model_path)

    def get_mapping_name(self):
        return f"bert_nsp_trained_mtl"
