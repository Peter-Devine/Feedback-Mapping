import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from mapping.mapping_models.mapping_models_base import BaseMapper

class TfidfPcaMapper(BaseMapper):

    def get_embeds(self):
        df = self.get_dataset(self.test_dataset, split="test")

        vectorizer = TfidfVectorizer()
        tf_idf_embed = vectorizer.fit_transform(df.text)
        pca = TruncatedSVD(n_components=500)
        low_dim_embed = pca.fit_transform(tf_idf_embed)

        return low_dim_embed, df.label

    def get_mapping_name(self, test_dataset):
        return "tfidf_pca"