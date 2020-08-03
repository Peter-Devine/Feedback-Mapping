from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from mapping.mapping_models.mapping_models_base import BaseMapper
from utils.utils import get_random_seed

class LdaMapper(BaseMapper):

    def get_embeds(self):
        df = self.get_dataset(dataset_name=self.test_dataset, app_name=self.app_name)

        # We remove stopwords and lowercase
        vectorizer = CountVectorizer(stop_words="english", lowercase=True)
        bow_embed = vectorizer.fit_transform(df.text.str.lower())

        lda = LatentDirichletAllocation(n_components=self.get_embedding_size(), max_iter=5, learning_method='online', learning_offset=50., random_state=get_random_seed())
        embeddings = lda.fit_transform(bow_embed)

        return embeddings, df

    def get_embedding_size(self):
        return 768

    def get_mapping_name(self):
        return f"lda_{self.get_embedding_size()}"
