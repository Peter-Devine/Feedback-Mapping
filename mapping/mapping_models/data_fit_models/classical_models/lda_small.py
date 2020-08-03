from mapping.mapping_models.data_fit_models.classical_models.lda import LdaMapper

class LdaSmallMapper(LdaMapper):

    def get_embedding_size(self):
        return 30
