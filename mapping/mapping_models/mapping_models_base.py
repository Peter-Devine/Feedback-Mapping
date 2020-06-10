import os
import pandas as pd

class BaseMapper:
    def __init__(self, test_task):
        self.test_task = test_task
        self.overall_dataset_dir = os.path.join(".", "data")

        # Raw dir
        self.raw_dataset_dir = os.path.join(self.overall_dataset_dir, "raw")
        assert os.path.exists(self.raw_dataset_dir), f"No raw datasets in the {self.raw_dataset_dir} path. Run data downloaders first."

        # Output dir
        self.output_dataset_dir = os.path.join(self.overall_dataset_dir, "embeddings")
        self.check_dir(self.output_dataset_dir)

        # Get mapping name
        self.mapping_name = self.get_mapping_name()

    def check_dir(self, dir):
        # Create directory if it doesn't exist
        if not os.path.exists(dir):
            print(f"Creating output directory at {dir}")
            os.mkdir(dir)

    def get_mapping_name(self):
        raise NotImplementedError

    def get_dataset(self, dataset, split=None):
        # Clean splits input
        splits = ["train", "val", "test"]
        assert split in splits, f"Unsupported split selected ({split}) \n{splits} selected"

        # Get the path of this dataset
        dataset_path = os.path.join(self.raw_dataset_dir, dataset, f"{split}.csv")

        # Return the df of this csv dataset
        return pd.read_csv(dataset_path)

    def output_embeddings(self, embedding, labels):
        # Make sure the embedding folder exists
        dataset_embedding_folder = os.path.join(self.output_dataset_dir, self.test_task)
        self.check_dir(dataset_embedding_folder)

        # Output the embeddings to a csv file in the embeddings folder
        dataset_embedding_file = os.path.join(dataset_embedding_folder, f"{self.mapping_name}.csv")

        # Get Dataframe from embeddings array
        embedding_df = pd.DataFrame(embedding)

        embedding_df["label"] = labels

        embedding_df.to_csv(dataset_embedding_file, index=False)

    def embed(self):
        embeddings, labels = self.get_embeds()

        self.output_embeddings(embeddings, labels)
