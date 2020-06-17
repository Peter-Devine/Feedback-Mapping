import math

import pandas as pd

import os

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel
from mapping.mapping_models.mapping_models_base import BaseMapper

class BertClsTrainedMapper(BaseMapper):

    def get_embeds(self):
        df = self.get_dataset(self.test_dataset, split="test")

        self.model_name = 'binwang/bert-base-nli'

        # Load pre-trained model tokenizer (vocabulary)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        # Load the BERT model
        model = self.get_model()
        model = model.to(self.device)
        model.eval()
        model.zero_grad()

        MAX_LENGTH = 128
        BATCH_SIZE = 64

        # Tokenize and convert to input IDs
        tokens_tensor = tokenizer.batch_encode_plus(list(df.text.values), max_length = MAX_LENGTH, pad_to_max_length=True, return_tensors="pt")
        tokens_tensor = tokens_tensor["input_ids"]

        # Create list for all embeddings to be saved to
        embeddings = []

        # Batch tensor so we can iterate over inputs
        test_loader = torch.utils.data.DataLoader(tokens_tensor, batch_size=BATCH_SIZE, shuffle=False)

        # Make sure the torch algorithm runs without gradients (as we aren't training)
        with torch.no_grad():
            print(f"Iterating over inputs {self.model_name} vanilla")
            # Iterate over all batches, passing the batches through the
            for test_batch in tqdm(test_loader):
                # See the models docstrings for the detail of the inputs
                outputs = model(test_batch.to(self.device))
                # Output the final average encoding across all characters as a numpy array
                np_array = outputs[0].mean(dim=1).numpy()
                # Append this encoding to a list
                embeddings.append(np_array)

        all_embeddings = np.concatenate(embeddings, axis=0)

        return all_embeddings, df.label

    def get_model(self):
        model_path = os.path.join(self.model_dir, f"{self.test_dataset}.pt")

        model = AutoModel.from_pretrained(self.model_name)

        if not os.path.exists(model_path):
            self.train_model(model_path)

        model.load_state_dict(torch.load(model_path))

        return model

    def train_model(self, model_path):
        df = self.get_dataset(self.test_dataset, split="test")

        

    def get_mapping_name(self, test_dataset):
        return f"bert_{test_dataset}cls_trained"
