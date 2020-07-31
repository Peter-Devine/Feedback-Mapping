import os
import pandas as pd

import torch
from transformers import AutoModel

from utils.utils import create_dir


class BaseMapper:
    def __init__(self, test_dataset):
        self.test_dataset = test_dataset
        self.overall_dataset_dir = os.path.join(".", "data")

        # Raw dir - the umbrella folder where we keep our initially downloaded data here
        self.raw_dataset_dir = os.path.join(self.overall_dataset_dir, "raw")
        assert os.path.exists(self.raw_dataset_dir), f"No raw datasets in the {self.raw_dataset_dir} path. Run data downloaders first."

        # Output dir - We keep our embeddings here
        self.output_dataset_dir = os.path.join(self.overall_dataset_dir, "embeddings")
        create_dir(self.output_dataset_dir)

        # Preprocessed dir
        self.preprocessed_dataset_dir = os.path.join(self.overall_dataset_dir, "preprocessed")
        create_dir(self.preprocessed_dataset_dir)

        # Auxiliary data dir - We keep extra data that we will not use for testing but that is used for training models
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

    def get_dataset(self, dataset_name, app_name):
        # Get the path of this dataset
        dataset_path = os.path.join(self.raw_dataset_dir, dataset_name, f"{app_name}.csv")

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

    def get_all_datasets(self):
        all_datasets = {}

        # Iterate over each folder in the raw data folder to find distinct datasets
        for item in os.listdir(self.raw_dataset_dir):
            data_folder_contents = os.path.join(self.raw_dataset_dir, item)

            # We make sure that this is a folder, and hence a dataset containing app.csv files
            if os.path.isdir(data_folder_contents):
                # We prepare a dict for this dataset
                all_datasets[item] = {}

                # Iterate over each app in the dataset
                for app_name in os.listdir(data_folder_contents):

                    # Get the df for this dataset and app, saving it to the all_datasets dict which will contain the data for every app in every dataset
                    all_datasets[item][app_name] = self.get_dataset(item, app_name)

        # Return the df of all datasets in raw folder
        return all_datasets

    def output_embeddings(self, embedding, df):
        # Make sure the embedding folder exists
        dataset_embedding_folder = os.path.join(self.output_dataset_dir, self.test_dataset)
        create_dir(dataset_embedding_folder)
        app_embedding_folder = os.path.join(dataset_embedding_folder, self.app_name)
        create_dir(app_embedding_folder)

        # Output the embeddings to a csv file in the embeddings folder
        app_embedding_file = os.path.join(app_embedding_folder, f"{self.mapping_name}.csv")

        # Get Dataframe from embeddings array
        embedding_df = pd.DataFrame(embedding, index=df.index)

        # Add labels to embeddings from passed df
        embedding_df["label"] = df["label"]

        embedding_df.to_csv(app_embedding_file)

    def embed(self):
        print(f"Now embedding all apps in {self.test_dataset} using {self.mapping_name}")

        # Iterate over each folder in the dataset to find all apps within it
        for app_file_name in os.listdir(self.raw_dataset_specific_dir):
            assert app_file_name[-4:] == ".csv", f"{app_file_name} file found in {self.raw_dataset_specific_dir} is not .csv. Please delete to run embedding on {self.test_dataset}"

            # Get the app name sans suffix. Save it as an attribute
            self.app_name = app_file_name[:-4]

            # Run the embedding for this app
            embedding_data = self.get_embeds()

            # If the embedding_data is None, which happens when a MISC_APP.csv dataset gets used on an app specific mapper, we simply skip over it.
            if embedding_data is None:
                print(f"Skipping embedding {self.test_dataset} ({self.app_name}) on {self.mapping_name}")
            else:
                embeddings, labels = embedding_data
                self.output_embeddings(embeddings, labels)
