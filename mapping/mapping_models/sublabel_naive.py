import numpy as np
from mapping.mapping_models.mapping_models_base import BaseMapper

class SublabelNaiveMapper(BaseMapper):

    def get_embeds(self):
        test_df = self.get_dataset(self.test_dataset, split="test")

        target_sublabels = test_df["sublabel1"].unique()
        def sublabel_embedder(input_sublabel):
            return np.array([1 if target_sublabel==input_sublabel else 0 for target_sublabel in target_sublabels])

        all_embeddings = test_df.sublabel1.apply(sublabel_embedder).values

        all_embeddings = np.stack(all_embeddings)

        return all_embeddings, test_df.label

    def get_mapping_name(self):
        return "sublabel_naive"
