import re
import zipfile
import requests
import io
import os
import numpy as np

from tqdm import tqdm

from mapping.mapping_models.mapping_models_base import BaseMapper

class GloveMapper(BaseMapper):

    def get_embeds(self):
        df = self.get_dataset(self.test_dataset, split="test")

        embedding_size = 300

        word_vector_file = os.path.join(self.model_dir, f"glove.6B.{embedding_size}d.txt")

        if not os.path.exists(word_vector_file):
            print("Downloading Glove...")
            r = requests.get("http://nlp.stanford.edu/data/glove.6B.zip")
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(path = self.model_dir)

        embeddings_dict = self.get_word_vector_dict(word_vector_file)

        def get_sentence_embedding(sentence):
            words = re.split('\W+', sentence)

            embedding = []

            for word in words:
                word_lower = word.lower()
                if word_lower in embeddings_dict.keys():
                    embedding.append(embeddings_dict[word_lower])

            if len(embedding) < 1:
                embedding = [np.asarray([0]*embedding_size)]

            return np.stack(embedding).mean(axis=0)

        tqdm.pandas()

        embeddings = df.text.progress_apply(get_sentence_embedding).values

        embeddings = np.stack(embeddings)

        return embeddings, df

    def get_word_vector_dict(self, file_name):
        embeddings_dict = {}
        with open(file_name, 'r', encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector

        return embeddings_dict

    def get_mapping_name(self):
        return "glove"
