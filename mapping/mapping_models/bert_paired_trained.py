import math

import pandas as pd

import os

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel
from mapping.mapping_models.mapping_models_base import BaseMapper
from mapping.model_training.transformer_training import train_nsp
from mapping.model_training.training_data_utils import shuffle_paired_df
from utils.utils import get_random_seed

class BertPairedTrainedMapper(BaseMapper):

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

    def get_auxiliary_dataset(self):
        auxiliary_dataset_path = os.path.join(self.auxiliary_dataset_dir, "github_issues.csv")

        if not os.path.exists(auxiliary_dataset_path):
            r = requests.get("https://storage.googleapis.com/kaggle-data-sets/10116%2F14254%2Fcompressed%2Fgithub_issues.csv.zip?GoogleAccessId=gcp-kaggle-com@kaggle-161607.iam.gserviceaccount.com&Expires=1593838591&Signature=n2Z%2BLOhR1yJq8%2Fe9DeoqpDwYx%2Fd0%2BeLgVADqEHAvLJGARv1rDQeW90ToP8J6IECtbLzGChwg0O18AjUUwbeNM70o3%2Bh32ej5lkl0RR3e89oAOP0IFMyL5JiRRFSrN%2ByobxbpIVLvz2R31qxIgpUp8DcQYDRaMvdIAdUEdXVUZBUqWuadFG08vtmwhWcQtL0gFlUOcsrrC2BCR3wWCiTPoQYouNEc0%2BXa13VTZlKeLW66R%2BZEG%2BOe0uxM%2BhZzZHsA7dA10tw23fBNHdsr5%2FITLLNu2y79QC8rFqrq3VxHIfaLT3aq%2Fyte3rr0TVSwwc43CMH9i35ibeS4N703zNaXjQ%3D%3D")
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(path = self.auxiliary_dataset_dir)

        # Read the csv file, rename the columns to standardised names and only take a subset of the data
        aux_df = pd.read_csv(auxiliary_dataset_path)
        aux_df = aux_df.rename({"issue_url": "id", "issue_title": "first_text", "body": "second_text"})
        aux_train_df = aux_df.sample(n=50000, random_state = get_random_seed())
        aux_val_df = aux_df.sample(n=50000, random_state = get_random_seed())

        # Make the data such that we have 50% paired and 50% unpaired
        paired_train_df = shuffle_paired_df(aux_train_df)
        paired_val_df = shuffle_paired_df(aux_val_df)

        return paired_train_df, paired_val_df

    def get_model(self):
        model_path = os.path.join(self.model_dir, f"{self.test_dataset}.pt")

        self.set_parameters()

        model = AutoModel.from_pretrained(self.model_name)

        if not os.path.exists(model_path):
            print(f"Running BERT paired training")
            self.train_model(model_path)


        print(f"Loading BERT paired model")
        model.load_state_dict(torch.load(model_path, map_location=self.device))

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

        model = train_nsp(train_df, valid_df, self.model_name, self.batch_size, self.max_length, self.device, params)

        torch.save(model.state_dict(), model_path)

    def get_mapping_name(self, test_dataset):
        return f"bert_paired_trained"
