import pandas as pd

import os
import requests
import tarfile
import shutil
import json

import torch
from transformers import AutoTokenizer, AutoModel
from mapping.mapping_models.mapping_models_base import BaseMapper
from mapping.model_training.transformer_training import train_cls
from mapping.model_training.training_data_utils import randomly_split_df, shuffle_paired_df, get_shuffled_second_text
from utils.utils import get_random_seed, bad_char_del
from utils.bert_utils import get_lm_embeddings

class BertPairedTrainedAltMapper(BaseMapper):

    def get_embeds(self):
        test_df = self.get_dataset(self.test_dataset, split="test")

        all_embeddings = get_lm_embeddings(self, test_df, f"{self.get_mapping_name()}")

        return all_embeddings, test_df

    def set_parameters(self):
        self.model_name = 'bert-base-uncased'
        self.max_length = 128
        self.batch_size = 32

    def get_auxiliary_dataset(self):

        target_zip_file_dir = os.path.join(self.auxiliary_dataset_dir, f"aux_alt_dataset")

        def download_project_data():
            # Download the paired bug dataset by https://github.com/whystar/MSR2018-DupPR
            url = f"https://github.com/whystar/MSR2018-DupPR/raw/master/dataset/raw.zip"

            if not os.path.exists(target_zip_file_dir):
                r = requests.get(url)
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall(path=target_zip_file_dir)

        # Download data as necessary
        if not os.path.exists(target_zip_file_dir):
            # N.B. we need to remove the "_article" substring when downloading for eclipse only
            download_project_data()

        # Get a df of the ids of duplicated issues
        r = requests.get("https://raw.githubusercontent.com/whystar/MSR2018-DupPR/master/dataset/duplicate.sql")
        data = r.content
        list_str = ", ".join([x.replace("INSERT INTO `duplicate` VALUES ", "").replace(";", "") for x in data.decode("utf-8").split("\n")[34:-4]])
        dup_df = pd.DataFrame([x[1:-1].split("', '") for x in list_str[2:-2].split("), (")], columns=["prj_id", "mst_pr", "dup_pr", "idn_cmt"])

        # Get a df of all pull requests on file
        with open(os.path.join(target_zip_file_dir, "raw", "pull-request.sql"), encoding="utf-8") as f:
            data = f.readlines()
        data = [x.replace("INSERT INTO `pull-request` VALUES ('", "")[:-3] for x in data if "INSERT INTO `pull-request` VALUES " in x]
        rows = []
        [rows.extend(x.split("'), ('")) for x in data]
        cells = [row.split("', '") for row in rows]
        text_df = pd.DataFrame(cells, columns=["id", "prj_id", "pr_num", "title", "description", "author", "created_at"])

        first_text = pd.merge(dup_df, text_df, how='inner', left_on=["prj_id", "mst_pr"], right_on=["prj_id", "pr_num"], suffixes=("_first", "_second"))
        paired_df = pd.merge(first_text, text_df, how='inner', left_on=["prj_id", "dup_pr"], right_on=["prj_id", "pr_num"], suffixes=("_first", "_second"))

        paired_df = paired_df.rename(columns = {"title_first": "first_text", "title_second": "second_text"})
        paired_df["label"] = 1

        first_unpaired_text, second_unpaired_text = get_shuffled_second_text(paired_df.first_text)
        unpaired_df = pd.DataFrame({"first_text": first_unpaired_text, "second_text": second_unpaired_text, "label": 0})
        all_paired_df = paired_df.append(unpaired_df).reset_index(drop=True)

        paired_train_df = all_paired_df.sample(frac=0.7, random_state = get_random_seed())
        paired_val_df = all_paired_df.drop(paired_train_df.index)

        shutil.rmtree(target_zip_file_dir)

        return paired_train_df, paired_val_df

    def get_model(self):
        model_path = os.path.join(self.model_dir, f"paired.pt")

        model = self.read_or_create_model(model_path)

        return model

    def train_model(self, model_path):
        train_df, valid_df = self.get_auxiliary_dataset()

        self.save_preprocessed_df(train_df, f"{self.test_dataset}_train")
        self.save_preprocessed_df(valid_df, f"{self.test_dataset}_val")

        training_data_dict = {self.test_dataset: (train_df, valid_df)}

        params = {
            "lr": 5e-5,
            "eps": 1e-6,
            "wd": 0.01,
            "epochs": 1,
            "patience": 2
        }

        model = train_cls(training_data_dict, self.model_name, self.batch_size, self.max_length, self.device, params, training_type="sim_cls")

        torch.save(model.state_dict(), model_path)

    def get_mapping_name(self):
        return f"bert_paired_trained_alt"
