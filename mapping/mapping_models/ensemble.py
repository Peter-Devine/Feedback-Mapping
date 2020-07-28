from clustering.clusterer import get_embeddings_paths, get_embedding_data

import numpy as np

from mapping.mapping_models.mapping_models_base import BaseMapper

class EnsembleMapper(BaseMapper):

    def get_embeds(self):
        embedding_files_data = get_embeddings_paths(self.test_dataset)

        # Go through all the available embeddings, and if we have some "gold standard" embeddings available, then we will just use those
        good_embeddings = ["sbert", "use", "bert_cls_trained_sublabel_mtl_test"]
        do_only_good_embeddings = False
        for embedding_file, embedding_name in embedding_files_data:
            if embedding_name in good_embeddings:
                do_only_good_embeddings = True

        concatenated_embedding = None
        for embedding_file, embedding_name in embedding_files_data:

            if do_only_good_embeddings and embedding_name not in good_embeddings:
                continue

            embeddings, labels = get_embedding_data(embedding_file)

            # We normalize embeddings so that the average magnitude of vectors is 1
            embeddings = embeddings / (np.linalg.norm(embeddings, axis=1).mean())

            if concatenated_embedding is None:
                concatenated_embedding = embeddings
            else:
                concatenated_embedding = np.concatenate([concatenated_embedding,embeddings], axis=1)

        # When get_embeds() is called, a numpy array of embeddings and a dataframe with label data is expected, so we pass labels as a df
        labels = pd.DataFrame({"label": labels})

        return concatenated_embedding, labels

    def get_mapping_name(self):
        return f"ensemble"
