import numpy as np
from mapping.mapping_models.mapping_models_base import BaseMapper

class KeywordMatchingMapper(BaseMapper):

    def get_embeds(self):
        test_df = self.get_dataset(dataset_name=self.test_dataset, app_name=self.app_name)

        keywords = ["bug", "crash", "feature", "app", "great", "new", "freeze", "free", "good"]
        def keyword_embedder(text):
            return np.array([1 if keyword in text.lower() else 0 for keyword in keywords])

        all_embeddings = test_df.text.apply(keyword_embedder).values

        all_embeddings = np.stack(all_embeddings)

        return all_embeddings, test_df

    def get_mapping_name(self):
        return "keyword_matching"
