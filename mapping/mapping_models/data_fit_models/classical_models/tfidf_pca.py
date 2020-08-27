from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from nltk.corpus import stopwords

from mapping.mapping_models.mapping_models_base import BaseMapper

class TfidfPcaMapper(BaseMapper):

    def get_embeds(self):
        df = self.get_dataset(dataset_name=self.test_dataset, app_name=self.app_name)

        # We remove stopwords and lowercase
        vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)

        text_series = df.text

        tf_idf_embed = vectorizer.fit_transform(text_series)

        # Get the smaller out of the chosen embedding size and the number of observations (cannot have more features than observations for SVD)
        embed_size = min([self.get_embedding_size(), df.shape[0]])
        pca = PCA(n_components=embed_size)
        low_dim_embed = pca.fit_transform(tf_idf_embed)

        return low_dim_embed, df

    def get_embedding_size(self):
        return 768

    def get_mapping_name(self):
        return f"tfidf_pca_{self.get_embedding_size()}"
