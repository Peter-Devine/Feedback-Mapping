import os
import pandas as pd

import torch

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

        # Get mapping name
        self.mapping_name = self.get_mapping_name(test_dataset)

        # Get the device that is available currently for torch training/inference
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model directory
        self.model_repo_dir = os.path.join(".", "mapping", "mapping_models", "saved_models")
        create_dir(self.model_repo_dir)
        self.model_dir = os.path.join(self.model_repo_dir, self.mapping_name)
        create_dir(self.model_dir)

    def get_mapping_name(self, dataset_name):
        raise NotImplementedError(f"get_mapping_name not implemented when running on {dataset_name} dataset")

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
        proprocessed_mapping_dir = os.path.join(self.preprocessed_dataset_dir, self.mapping_name)
        create_dir(proprocessed_mapping_dir)
        df.to_csv(os.path.join(proprocessed_mapping_dir, f"{filename}.csv"))

    def get_all_datasets(self, split):
        all_datasets = {}

        # Iterate over each folder in the raw data folder to find distinct datasets
        for item in os.listdir(self.raw_dataset_dir):
            if os.path.isdir(os.path.join(self.raw_dataset_dir, item)) and item[0] != ".":
                all_datasets[item] = self.get_dataset(item, split)

        # Return the df of all datasets in raw folder
        return all_datasets

    def output_embeddings(self, embedding, labels):
        # Make sure the embedding folder exists
        dataset_embedding_folder = os.path.join(self.output_dataset_dir, self.test_dataset)
        create_dir(dataset_embedding_folder)

        # Output the embeddings to a csv file in the embeddings folder
        dataset_embedding_file = os.path.join(dataset_embedding_folder, f"{self.mapping_name}.csv")

        # Get Dataframe from embeddings array
        embedding_df = pd.DataFrame(embedding, index=labels.index)

        embedding_df["label"] = labels

        embedding_df.to_csv(dataset_embedding_file)

    def embed(self):
        print(f"Now embedding {self.test_dataset} using {self.mapping_name}")

        embeddings, labels = self.get_embeds()

        self.output_embeddings(embeddings, labels)
