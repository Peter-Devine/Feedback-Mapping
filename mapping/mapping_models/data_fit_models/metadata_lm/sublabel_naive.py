import numpy as np
from mapping.mapping_models.mapping_models_base import BaseMapper

class SublabelNaiveMapper(BaseMapper):

    def get_embeds(self):
        test_df = self.get_dataset(dataset_name=self.test_dataset, app_name=self.app_name)

        target_sublabels = test_df["sublabel"].unique()
        def sublabel_embedder(input_sublabel):
            return np.array([1 if target_sublabel==input_sublabel else 0 for target_sublabel in target_sublabels])

        all_embeddings = test_df.sublabel.apply(sublabel_embedder).values

        all_embeddings = np.stack(all_embeddings)

        return all_embeddings, test_df

    def get_mapping_name(self):
        return "sublabel_naive"
