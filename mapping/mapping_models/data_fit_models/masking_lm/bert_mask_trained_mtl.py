import os

from mapping.mapping_models.mapping_models_base import BaseMapper
from utils.bert_utils import get_lm_embeddings

class BertMaskTrainedMtlMapper(BaseMapper):

    def get_embeds(self):
        test_df = self.get_dataset(dataset_name=self.test_dataset, app_name=self.app_name)

        all_embeddings = get_lm_embeddings(self, test_df, f"{self.get_mapping_name()}")

        return all_embeddings, test_df

    def set_parameters(self):
        self.model_name = 'bert-base-uncased'
        self.max_length = 128
        self.batch_size = 64
        self.lr = 5e-5
        self.eps = 1e-6
        self.wd = 0.01
        self.epochs = 30
        self.patience = 1

    def get_model(self):
        model_path = os.path.join(self.model_dir, self.get_model_name())

        model = self.read_or_create_model(model_path)

        return model

    def get_model_name(self):
        return "all.pt"

    def get_training_data(self):
        return self.get_all_datasets()

    def train_model(self, model_path):
        train_dfs = self.get_training_data()

        combined_df = self.prepare_train_tasks_dataset(train_dfs)

        params = {
            "lr": self.lr,
            "eps": self.eps,
            "wd": self.wd,
            "epochs": self.epochs,
            "patience": self.patience,
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
        }

        model = train_cls(mtl_datasets, params, self.device)

        torch.save(model.state_dict(), model_path)

    def prepare_train_tasks_dataset(self, train_dfs_dict):
        # Combine all app data for all datasets into a single df
        combined_df = None

        for dataset_name in train_dfs_dict.keys():
            # Get the df for a given dataset
            app_train_dfs_dict = train_dfs_dict[dataset_name]

            for app_name, app_df in app_train_dfs_dict.items():
                if combined_df is None:
                    combined_df = app_df
                else:
                    combined_df = combined_df.append(app_df).reset_index(drop=True)

        # Save this df for debugging purposes
        self.save_preprocessed_df(combined_df, f"{dataset_name}_{self.app_name}")

        return combined_df

    def get_mapping_name(self):
        return f"bert_mask_trained_mtl"
