import math

import pandas as pd

import os
import requests
import zipfile
import io

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel
from mapping.mapping_models.mapping_models_base import BaseMapper
from mapping.model_training.transformer_training import train_sim
from mapping.model_training.training_data_utils import shuffle_paired_df
from utils.utils import get_random_seed

class BertPairedTrainedMapper(BaseMapper):

    def get_embeds(self):
        test_df = self.get_dataset(self.test_dataset, split="test")

        all_embeddings = get_lm_embeddings(self, test_df, f"{self.get_mapping_name()}")

        return all_embeddings, test_df.label

    def set_parameters(self):
        self.model_name = 'bert-base-uncased'
        self.max_length = 128
        self.batch_size = 32

    def get_auxiliary_dataset(self):
        auxiliary_dataset_path = os.path.join(self.auxiliary_dataset_dir, "github_issues.csv")

        if not os.path.exists(auxiliary_dataset_path):
            r = requests.get("https://storage.googleapis.com/kaggle-data-sets/10116%2F14254%2Fcompressed%2Fgithub_issues.csv.zip?GoogleAccessId=gcp-kaggle-com@kaggle-161607.iam.gserviceaccount.com&Expires=1593838591&Signature=n2Z%2BLOhR1yJq8%2Fe9DeoqpDwYx%2Fd0%2BeLgVADqEHAvLJGARv1rDQeW90ToP8J6IECtbLzGChwg0O18AjUUwbeNM70o3%2Bh32ej5lkl0RR3e89oAOP0IFMyL5JiRRFSrN%2ByobxbpIVLvz2R31qxIgpUp8DcQYDRaMvdIAdUEdXVUZBUqWuadFG08vtmwhWcQtL0gFlUOcsrrC2BCR3wWCiTPoQYouNEc0%2BXa13VTZlKeLW66R%2BZEG%2BOe0uxM%2BhZzZHsA7dA10tw23fBNHdsr5%2FITLLNu2y79QC8rFqrq3VxHIfaLT3aq%2Fyte3rr0TVSwwc43CMH9i35ibeS4N703zNaXjQ%3D%3D")
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(path = self.auxiliary_dataset_dir)

        # Read the csv file, rename the columns to standardised names and only take a subset of the data
        aux_df = pd.read_csv(auxiliary_dataset_path)
        # We aim to pair titles with bodies for a large number of GitHub issues
        aux_df = aux_df.rename(columns={"issue_url": "id", "issue_title": "first_text", "body": "second_text"})
        aux_train_df = aux_df.sample(n=100000, random_state = get_random_seed())
        aux_val_df = aux_df.sample(n=1000, random_state = get_random_seed())

        # Make the data such that we have 50% paired and 50% unpaired
        paired_train_df = shuffle_paired_df(aux_train_df)
        paired_val_df = shuffle_paired_df(aux_val_df)

        return paired_train_df, paired_val_df

    def get_model(self):
        model_path = os.path.join(self.model_dir, f"paired.pt")

        model = self.read_or_create_model(model_path)

        return model

    def train_model(self, model_path):
        train_df, valid_df = self.get_auxiliary_dataset()

        self.save_preprocessed_df(train_df, f"{self.test_dataset}_train")
        self.save_preprocessed_df(valid_df, f"{self.test_dataset}_val")

        params = {
            "lr": 5e-5,
            "eps": 1e-6,
            "wd": 0.01,
            "epochs": 1,
            "patience": 2
        }

        model = train_sim(train_df, valid_df, self.model_name, self.batch_size, self.max_length, self.device, params)

        torch.save(model.state_dict(), model_path)

    def get_mapping_name(self, test_dataset):
        return f"bert_paired_trained"
