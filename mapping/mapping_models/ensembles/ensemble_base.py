import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from mapping.mapping_models.mapping_models_base import BaseMapper

class EnsembleBaseMapper(BaseMapper):

    def get_embeds(self):

        embeddings_dir = os.path.join(".", "data", "embeddings", self.test_dataset, self.app_name)
        raw_dir = os.path.join(".", "data", "raw", self.test_dataset, f"{self.app_name}.csv")

        raw_df = pd.read_csv(raw_dir)

        concatenated_embedding = None

        # Cycle through all the embedding types given for this ensembler
        embedding_types_to_ensemble = self.get_ensemble_components()
        for embedding_type in embedding_types_to_ensemble:
            # Get the dir for this embedding, and make sure it exists first
            embedding_dir = os.path.join(embeddings_dir, f"{embedding_type}.csv")
            assert os.path.exists(embedding_dir), f"Ensembler was asked to ensemble {embedding_types_to_ensemble} embedding types, but {embedding_dir} does not exist."

            # Read the embedding
            embedding_df = pd.read_csv(embedding_dir, index_col = 0)

            # Get gold label for each embedding
            labels = embedding_df["label"]

            # Get embedding values
            embeddings = embedding_df.drop("label", axis=1)

            # We normalize embeddings so that the average magnitude of vectors is 1
            embeddings = embeddings / (np.linalg.norm(embeddings, axis=1).mean())

            if concatenated_embedding is None:
                concatenated_embedding = embeddings
            else:
                concatenated_embedding = np.concatenate([concatenated_embedding, embeddings], axis=1)

        # Reduce size of ensemble embedding if set
        PCA_SIZE = self.get_pca_size(concatenated_embedding)
        if PCA_SIZE > 0:
            pca = PCA(n_components=PCA_SIZE)
            concatenated_embedding = pca.fit_transform(concatenated_embedding)

        return concatenated_embedding, raw_df

    def get_pca_size(self, concatenated_embedding):
        # Returns the number of components to use in PCA decomposition.
        # 0 for no PCA decomposition.
        return 0

    def get_ensemble_components(self):
        raise NotImplementedError(f"Ensemble mapper needs a list of embeddings with which to create the ensemble")

    def get_mapping_name(self):
        embeddings_string = "__".join(self.get_ensemble_components())
        return f"ensemble__{embeddings_string}_{get_pca_size()}"
