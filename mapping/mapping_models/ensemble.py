from clustering.clusterer import get_embeddings_paths, get_embedding_data

import numpy as np

from mapping.mapping_models.mapping_models_base import BaseMapper

class EnsembleMapper(BaseMapper):

    def get_embeds(self):
        embedding_files_data = get_embeddings_paths(self.test_dataset)

        concatenated_embedding = None
        for embedding_file, embedding_name in embedding_files_data:

            embeddings, labels = get_embedding_data(embedding_file)

            if concatenated_embedding is None:
                concatenated_embedding = embeddings
            else:
                concatenated_embedding = np.concatenate([concatenated_embedding,embeddings], axis=1)

        return concatenated_embedding, labels

    def get_mapping_name(self):
        return f"ensemble"
