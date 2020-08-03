from mapping.mapping_models.data_fit_models.classical_models.tfidf_pca import TfidfPcaMapper

class TfidfPcaSmallMapper(TfidfPcaMapper):

    def get_embedding_size(self):
        return 30
