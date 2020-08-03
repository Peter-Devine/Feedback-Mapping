from mapping.mapping_models.data_fit_models.classical_models.gsdmm import GsdmmMapper

class GsdmmSmallMapper(GsdmmMapper):

    def get_embedding_size(self):
        return 30
