import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
from mapping.mapping_models.mapping_models_base import BaseMapper

class UseMapper(BaseMapper):

    def get_embeds(self):
        df = self.get_dataset(self.test_task, split="test")

        embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        embeddings = embed(df["text"])

        return embeddings.numpy(), df.label

    def get_mapping_name(self):
        return "use"
