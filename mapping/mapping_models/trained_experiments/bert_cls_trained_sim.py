import pandas as pd
import os

from tqdm import tqdm

import torch
from transformers import AutoModel
from mapping.mapping_models.mapping_models_base import BaseMapper
from mapping.model_training.transformer_training import train_cls
from mapping.model_training.training_data_utils import get_cls_pair_matched_df
from utils.bert_utils import get_lm_embeddings

class BertClsTrainedSimMapper(BaseMapper):

    def get_embeds(self):
        test_df = self.get_dataset(self.test_dataset, split="test")

        all_embeddings = get_lm_embeddings(self, test_df, f"{self.get_mapping_name()}")

        return all_embeddings, test_df

    def set_parameters(self):
        self.model_name = 'bert-base-uncased'
        self.max_length = 128
        self.batch_size = 32

    def get_model(self):
        model_path = os.path.join(self.model_dir, f"{self.test_dataset}.pt")

        model = self.read_or_create_model(model_path)

        return model

    def train_model(self, model_path):
        # Get the training data for the given dataset
        train_df = self.get_dataset(self.test_dataset, split="train")
        valid_df = self.get_dataset(self.test_dataset, split="val")

        # Get df where a given text observation is eitehr paired with another observation of their class, or an observation of another class.
        # This difference is the label (1/0) and is what the similarity model is trying to predict
        train_df = get_cls_pair_matched_df(train_df, class_column="label")
        valid_df = get_cls_pair_matched_df(valid_df, class_column="label")

        training_data_dict = {self.test_dataset: (train_df, valid_df)}

        params = {
            "lr": 5e-5,
            "eps": 1e-6,
            "wd": 0.01,
            "epochs": 2+int(10000/train_df.shape[0]),
            "patience": 2
        }

        model = train_cls(training_data_dict, self.model_name, self.batch_size, self.max_length, self.device, params, training_type="sim_cls")

        torch.save(model.state_dict(), model_path)

    def get_mapping_name(self):
        return f"bert_cls_trained_sim"
