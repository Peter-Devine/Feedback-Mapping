import math

import pandas as pd

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel
from mapping.mapping_models.mapping_models_base import BaseMapper

class T5VanillaMapper(BaseMapper):

    def get_embeds(self):
        test_df = self.get_dataset(self.test_dataset, split="test")

        all_embeddings = get_lm_embeddings(self, test_df, f"{self.get_mapping_name()}")

        return all_embeddings, test_df.label

    def get_model(self):
        model = AutoModel.from_pretrained(self.model_name)
        model = model.encoder

        return model

    def set_parameters(self):
        self.model_name = 't5-small'
        self.max_length = 128
        self.batch_size = 64

    def get_mapping_name(self):
        return "t5_vanilla"
