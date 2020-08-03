import pandas as pd
import os

from tqdm import tqdm

import torch
from transformers import AutoModel
from mapping.mapping_models.mapping_models_base import BaseMapper
from mapping.model_training.transformer_training import train_cls
from utils.bert_utils import get_lm_embeddings
from utils.utils import get_random_seed

class BertClsTrainedSublabelOutsideMapper(BaseMapper):

    def get_embeds(self):
        test_df = self.get_dataset(self.test_dataset, split="test")

        all_embeddings = get_lm_embeddings(self, test_df, f"{self.get_mapping_name()}")

        return all_embeddings, test_df

    def set_parameters(self):
        self.model_name = 'bert-base-uncased'
        self.max_length = 128
        self.batch_size = 64

    def get_model(self):
        model_path = os.path.join(self.model_dir, f"all.pt")

        model = self.read_or_create_model(model_path)

        return model

    def get_auxiliary_dataset(self):
        # Use the review dataset gathered for https://giograno.me/assets/pdf/workshop/wama17.pdf
        df = pd.read_csv("https://raw.githubusercontent.com/sealuzh/user_quality/master/csv_files/reviews.csv", index_col=0)
        df = df.rename(columns={"review": "text", "star": "label", "package_name": "sublabel1"})

        train_df = df.sample(frac=0.7, random_state = get_random_seed())
        val_df = df.drop(train_df.index)
        return train_df, val_df

    def train_model(self, model_path):
        train_df, val_df = self.get_auxiliary_dataset()

        train_datasets = {"sublabel_outside": (train_df, val_df)}

        # Save this df for debugging purposes
        self.save_preprocessed_df(train_df, f"sublabel_outside_train")
        self.save_preprocessed_df(val_df, f"sublabel_outside_val")

        params = {
            "lr": 5e-5,
            "eps": 1e-6,
            "wd": 0.01,
            "epochs": 5,
            "patience": 1
        }

        model = train_cls(train_datasets, self.model_name, self.batch_size, self.max_length, self.device, params)

        torch.save(model.state_dict(), model_path)

    def get_mapping_name(self):
        return f"bert_cls_trained_sublabel_outside"
