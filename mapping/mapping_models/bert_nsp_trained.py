import math

import pandas as pd

import os

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel
from mapping.mapping_models.mapping_models_base import BaseMapper
from mapping.model_training.transformer_training import train_nsp
from mapping.model_training.training_data_utils import get_next_sentence_df

class BertNspTrainedMapper(BaseMapper):

    def get_embeds(self):
        df = self.get_dataset(self.test_dataset, split="test")

        self.set_parameters()

        # Load pre-trained model tokenizer (vocabulary)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        # Load the BERT model
        model = self.get_model()
        model = model.to(self.device)
        model.eval()
        model.zero_grad()

        # Tokenize and convert to input IDs
        tokens_tensor = tokenizer.batch_encode_plus(list(df.text.values), max_length = self.max_length, pad_to_max_length=True, truncation=True, return_tensors="pt")
        tokens_tensor = tokens_tensor["input_ids"]

        # Create list for all embeddings to be saved to
        embeddings = []

        # Batch tensor so we can iterate over inputs
        test_loader = torch.utils.data.DataLoader(tokens_tensor, batch_size=self.batch_size, shuffle=False)

        # Make sure the torch algorithm runs without gradients (as we aren't training)
        with torch.no_grad():
            print(f"Iterating over inputs {self.model_name} NSP trained")
            # Iterate over all batches, passing the batches through the
            for test_batch in tqdm(test_loader):
                # See the models docstrings for the detail of the inputs
                outputs = model(test_batch.to(self.device))
                # Output the final average encoding across all characters as a numpy array
                np_array = outputs[0].mean(dim=1).cpu().numpy()
                # Append this encoding to a list
                embeddings.append(np_array)

        all_embeddings = np.concatenate(embeddings, axis=0)

        return all_embeddings, df.label

    def set_parameters(self):
        self.model_name = 'bert-base-uncased'
        self.max_length = 128
        self.batch_size = 32

    def get_model(self):
        model_path = os.path.join(self.model_dir, f"{self.test_dataset}.pt")

        self.set_parameters()

        model = AutoModel.from_pretrained(self.model_name)

        if not os.path.exists(model_path):
            print(f"Running BERT NSP training on {self.test_dataset}")
            self.train_model(model_path)


        print(f"Loading BERT NSP model trained on {self.test_dataset}")
        model.load_state_dict(torch.load(model_path, map_location=self.device))

        return model

    def train_model(self, model_path):
        train_df = self.get_dataset(self.test_dataset, split="train")
        valid_df = self.get_dataset(self.test_dataset, split="val")

        train_df = get_next_sentence_df(train_df)
        valid_df = get_next_sentence_df(valid_df)

        self.save_preprocessed_df(train_df, "train")
        self.save_preprocessed_df(valid_df, "val")

        params = {
            "lr": 5e-5,
            "eps": 1e-6,
            "wd": 0.01,
            "epochs": int(10000/train_df.shape[0]),
            "patience": 2
        }

        model = train_nsp(train_df, valid_df, self.model_name, self.batch_size, self.max_length, self.device, params)

        torch.save(model.state_dict(), model_path)

    def get_mapping_name(self, test_dataset):
        return f"bert_nsp_trained"
