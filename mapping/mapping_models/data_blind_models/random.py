import numpy as np

from mapping.mapping_models.mapping_models_base import BaseMapper

class RandomMapper(BaseMapper):

    def get_embeds(self):
        df = self.get_dataset(dataset_name=self.test_dataset, app_name=self.app_name)

        num_observations = df.shape[0]
        desired_dim = 500

        embedding = np.random.rand(num_observations, desired_dim)

        return embedding, df

    def get_mapping_name(self):
        return "random"
