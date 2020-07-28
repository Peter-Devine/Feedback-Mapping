import os
import pandas as pd

import torch
from transformers import AutoModel

from utils.utils import create_dir


class BaseMapper:
    def __init__(self, test_dataset):
        self.test_dataset = test_dataset
        self.overall_dataset_dir = os.path.join(".", "data")

        # Raw dir
        self.raw_dataset_dir = os.path.join(self.overall_dataset_dir, "raw")
        assert os.path.exists(self.raw_dataset_dir), f"No raw datasets in the {self.raw_dataset_dir} path. Run data downloaders first."

        # Output dir
        self.output_dataset_dir = os.path.join(self.overall_dataset_dir, "embeddings")
        create_dir(self.output_dataset_dir)

        # Preprocessed dir
        self.preprocessed_dataset_dir = os.path.join(self.overall_dataset_dir, "preprocessed")
        create_dir(self.preprocessed_dataset_dir)

        # Preprocessed dir
        self.base_auxiliary_dataset_dir = os.path.join(self.overall_dataset_dir, "auxiliary")
        create_dir(self.base_auxiliary_dataset_dir)

        # Get mapping name
        self.mapping_name = self.get_mapping_name()

        # Get the device that is available currently for torch training/inference
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model directory
        self.model_repo_dir = os.path.join(".", "mapping", "mapping_models", "saved_models")
        create_dir(self.model_repo_dir)
        self.model_dir = os.path.join(self.model_repo_dir, self.mapping_name)
        create_dir(self.model_dir)
        self.auxiliary_dataset_dir = os.path.join(self.base_auxiliary_dataset_dir, self.mapping_name)
        create_dir(self.auxiliary_dataset_dir)

    def get_mapping_name(self):
        raise NotImplementedError(f"get_mapping_name not implemented when running on {self.test_dataset} dataset")

    def get_dataset(self, dataset, split=None):
        # Clean splits input
        splits = ["train", "val", "test"]
        assert split in splits, f"Unsupported split selected ({split}) \n{splits} selected"

        # Get the path of this dataset
        dataset_path = os.path.join(self.raw_dataset_dir, dataset, f"{split}.csv")

        assert os.path.exists(dataset_path), f"No file found at {dataset_path}"

        # Return the df of this csv dataset
        return pd.read_csv(dataset_path, index_col = 0)

    def save_preprocessed_df(self, df, filename):
        preprocessed_mapping_dir = os.path.join(self.preprocessed_dataset_dir, self.mapping_name)
        create_dir(preprocessed_mapping_dir)
        df.to_csv(os.path.join(preprocessed_mapping_dir, f"{filename}.csv"))

    def read_or_create_model(self, model_file_path, model=None):
        self.set_parameters()

        # If we have not been supplied with a model, then initialize one from the huggingface library
        if model is None:
            model = AutoModel.from_pretrained(self.model_name)

        if not os.path.exists(model_file_path):
            print(f"Running training to create {model_file_path}")
            self.train_model(model_file_path)


        print(f"Loading model from {model_file_path}")
        model.load_state_dict(torch.load(model_file_path, map_location=self.device))

        return model

    def get_all_datasets(self, split):
        all_datasets = {}

        # Iterate over each folder in the raw data folder to find distinct datasets
        for item in os.listdir(self.raw_dataset_dir):
            if os.path.isdir(os.path.join(self.raw_dataset_dir, item)) and item[0] != ".":
                all_datasets[item] = self.get_dataset(item, split)

        # Return the df of all datasets in raw folder
        return all_datasets

    def output_embeddings(self, embedding, df):
        # Make sure the embedding folder exists
        dataset_embedding_folder = os.path.join(self.output_dataset_dir, self.test_dataset)
        create_dir(dataset_embedding_folder)

        # Output the embeddings to a csv file in the embeddings folder
        dataset_embedding_file = os.path.join(dataset_embedding_folder, f"{self.mapping_name}.csv")

        # Get Dataframe from embeddings array
        embedding_df = pd.DataFrame(embedding, index=df.index)

        # Set the fine_label (I.e. the more fine-grained label which no model has been trained on) as the label if available. Else, set it as the training labels
        label_col = "fine_label" if "fine_label" in  df.columns else "label"
        embedding_df["label"] = df[label_col]

        embedding_df.to_csv(dataset_embedding_file)

    def embed(self):
        print(f"Now embedding {self.test_dataset} using {self.mapping_name}")

        embeddings, labels = self.get_embeds()

        self.output_embeddings(embeddings, labels)
