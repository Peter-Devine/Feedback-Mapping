from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from mapping.mapping_models.mapping_models_base import BaseMapper

class LdaMapper(BaseMapper):

    def get_embeds(self):
        df = self.get_dataset(self.test_dataset, split="test")

        vectorizer = CountVectorizer()
        bow_embed = vectorizer.fit_transform(df.text)

        lda = LatentDirichletAllocation(n_components=100, max_iter=5, learning_method='online', learning_offset=50., random_state=self.random_state)
        embeddings = lda.fit_transform(bow_embed)

        return embeddings, df.label

    def get_mapping_name(self, test_dataset):
        return "lda"
