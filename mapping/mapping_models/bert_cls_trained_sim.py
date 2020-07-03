import pandas as pd
import os

from tqdm import tqdm

import torch
from transformers import AutoModel
from mapping.mapping_models.mapping_models_base import BaseMapper
from mapping.model_training.transformer_training import train_sim
from mapping.model_training.training_data_utils import pair_class_df, shuffle_paired_df
from utils.bert_utils import get_lm_embeddings

class BertClsTrainedSimMapper(BaseMapper):

    def get_embeds(self):
        test_df = self.get_dataset(self.test_dataset, split="test")

        all_embeddings = get_lm_embeddings(self, test_df, f"{self.get_mapping_name()}")

        return all_embeddings, test_df.label

    def set_parameters(self):
        self.model_name = 'binwang/bert-base-nli'
        self.max_length = 128
        self.batch_size = 64

    def get_model(self):
        model_path = os.path.join(self.model_dir, f"{self.test_dataset}.pt")

        model = self.read_or_create_model(model_path)

        return model

    def train_model(self, model_path):
        # Get the training data for the given dataset
        train_df = self.get_dataset(self.test_dataset, split="train")
        valid_df = self.get_dataset(self.test_dataset, split="val")

        # Change this datatype into a df which contains pieces of text paired by class
        train_df = pair_class_df(train_df, class_column="label")
        valid_df = pair_class_df(valid_df, class_column="label")

        # Then randomly unpair half of the class-paired texts as the out-of-class examples for training
        train_df = shuffle_paired_df(train_df)
        valid_df = shuffle_paired_df(valid_df)

        params = {
            "lr": 5e-5,
            "eps": 1e-6,
            "wd": 0.01,
            "epochs": int(10000/train_df.shape[0]),
            "patience": 2
        }

        model = train_sim(train_df, valid_df, self.model_name, self.batch_size, self.max_length, self.device, params)

        torch.save(model.state_dict(), model_path)

    def get_mapping_name(self):
        return f"bert_cls_trained_sim"
