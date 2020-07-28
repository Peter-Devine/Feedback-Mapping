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
from mapping.model_training.training_data_utils import randomly_split_df, shuffle_paired_df
from utils.utils import get_random_seed, bad_char_del
from utils.bert_utils import get_lm_embeddings

class BertPairedTrainedMapper(BaseMapper):

    def get_embeds(self):
        test_df = self.get_dataset(self.test_dataset, split="test")

        all_embeddings = get_lm_embeddings(self, test_df, f"{self.get_mapping_name()}")

        return all_embeddings, test_df

    def set_parameters(self):
        self.model_name = 'bert-base-uncased'
        self.max_length = 128
        self.batch_size = 32

    def get_auxiliary_dataset(self):

        def download_project_data(filename):
            # Download the paired bug dataset by Irving Muller Rodrigues, Daniel Aloise, Eraldo Rezende Fernandes, and Michel Dagenais (Ref https://github.com/irving-muller/soft_alignment_model_bug_deduplication)
            url = f"https://zenodo.org/record/3922012/files/{filename}.tar.gz?download=1"
            target_tar_file_dir = os.path.join(self.auxiliary_dataset_dir, f"{filename}.tar.gz")

            if not os.path.exists(target_tar_file_dir):
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(target_tar_file_dir, 'wb') as f:
                        f.write(response.raw.read())
                else:
                    raise Exception(f"Response status code {response.status_code} when trying to download paired dataset from {url}")

            # Unzip downloaded tar file
            tar = tarfile.open(target_tar_file_dir, "r:gz")
            tar.extractall(path=self.auxiliary_dataset_dir)
            tar.close()

        def get_aux_paired_text_df(filename, split, del_files=False):
            # Download data as necessary
            unzipped_files_dir = os.path.join(self.auxiliary_dataset_dir, f"{filename}")
            if not os.path.exists(unzipped_files_dir):
                # N.B. we need to remove the "_article" substring when downloading for eclipse only
                download_project_data(filename.replace("_article", ""))

            if "eclipse" in filename:
                small_filename = "eclipse"
            elif "mozilla" in filename:
                small_filename = "mozilla"
            elif "netbeans" in filename:
                small_filename = "netbeans"
            elif "open_office" in filename:
                small_filename = "open_office"

            # Prepare a dataframe with all pair IDs and labels
            pair_file_dir = os.path.join(unzipped_files_dir, f"{split}_{small_filename}_pairs_random_1.txt")
            with open(pair_file_dir, 'r') as f:
                data = f.readlines()
            split_data = [x.split(",") for x in data]
            matching_df = pd.DataFrame([{"first_text_id": x[0], "second_text_id": x[1], "label": x[2].strip()} for x in split_data])

            # Prepare a dataframe with all text data
            text_file_dir = os.path.join(unzipped_files_dir, f"{small_filename}_initial.json")
            with open(text_file_dir,'r') as f:
                data = [json.loads(datum) for datum in f.readlines()]
            text_df = pd.DataFrame(data)

            if del_files:
                # Delete files now that they have been read and will not be used any more
                shutil.rmtree(unzipped_files_dir)

            # Merge these two dataframes to have the label, first text and second text on each row
            first_merged_df = pd.merge(matching_df, text_df, how='inner', left_on="first_text_id", right_on="bug_id")
            full_merged_df = pd.merge(first_merged_df, text_df, how='inner', left_on="second_text_id", right_on="bug_id", suffixes=("_first", "_second"))

            # The original dataset has labels of 1 and -1, so we fix that to make it 1 and 0
            full_merged_df["label"] = full_merged_df["label"].apply(lambda x: 1 if int(x) == 1 else 0)

            paired_df = full_merged_df[["short_desc_first", "short_desc_second", "label"]]
            paired_df = paired_df.rename(columns = {"short_desc_first": "first_text", "short_desc_second": "second_text"})

            paired_df["first_text"] = paired_df["first_text"].apply(bad_char_del)
            paired_df["second_text"] = paired_df["second_text"].apply(bad_char_del)

            return paired_df

        ec_train = get_aux_paired_text_df("eclipse_2001-2007_2008_article", "training").sample(n=10000, random_state = get_random_seed())
        ec_val = get_aux_paired_text_df("eclipse_2001-2007_2008_article", "validation").sample(n=2000, random_state = get_random_seed())
        mz_train = get_aux_paired_text_df("mozilla_2001-2009_2010", "training").sample(n=10000, random_state = get_random_seed())
        mz_val = get_aux_paired_text_df("mozilla_2001-2009_2010", "validation").sample(n=2000, random_state = get_random_seed())
        nb_train = get_aux_paired_text_df("netbeans_2001-2007_2008", "training").sample(n=10000, random_state = get_random_seed())
        nb_val = get_aux_paired_text_df("netbeans_2001-2007_2008", "validation").sample(n=2000, random_state = get_random_seed())
        oo_train = get_aux_paired_text_df("open_office_2001-2008_2010", "training").sample(n=10000, random_state = get_random_seed())
        oo_val = get_aux_paired_text_df("open_office_2001-2008_2010", "validation").sample(n=2000, random_state = get_random_seed())

        paired_train_df = ec_train.append(mz_train).append(nb_train).append(oo_train).reset_index(drop=True)
        paired_val_df = ec_val.append(mz_val).append(nb_val).append(oo_val).reset_index(drop=True)

        return paired_train_df, paired_val_df

    def get_model(self):
        model_path = os.path.join(self.model_dir, f"paired.pt")

        model = self.read_or_create_model(model_path)

        return model

    def train_model(self, model_path):
        train_df, valid_df = self.get_auxiliary_dataset()

        self.save_preprocessed_df(train_df, f"paired_train")
        self.save_preprocessed_df(valid_df, f"paired_val")

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
        return f"bert_paired_trained"
