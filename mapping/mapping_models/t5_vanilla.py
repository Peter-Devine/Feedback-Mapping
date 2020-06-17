import math

import pandas as pd

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel
from mapping.mapping_models.mapping_models_base import BaseMapper

class T5VanillaMapper(BaseMapper):

    def get_embeds(self):
        df = self.get_dataset(self.test_dataset, split="test")

        self.model_name = 't5-small'

        # Load pre-trained model tokenizer (vocabulary)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        # Load the BERT model
        model = AutoModel.from_pretrained(self.model_name)
        model = model.encoder
        model = model.to(self.device)

        MAX_LENGTH = 128
        BATCH_SIZE = 64

        # Tokenize and convert to input IDs
        tokens_tensor = tokenizer.batch_encode_plus(list(df.text.values), max_length = MAX_LENGTH, pad_to_max_length=True, return_tensors="pt")
        tokens_tensor = tokens_tensor["input_ids"]

        # Create list for all embeddings to be saved to
        embeddings = []

        # Get the number of observations to embed
        num_obs = tokens_tensor.shape[0]

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

    def get_mapping_name(self, test_dataset):
        return "t5_vanilla"
