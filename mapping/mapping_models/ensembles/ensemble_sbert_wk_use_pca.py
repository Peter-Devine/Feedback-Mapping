from mapping.mapping_models.ensembles.ensemble_base import EnsembleBaseMapper

class EnsembleSbertWkUsePcaMapper(EnsembleBaseMapper):

    def get_ensemble_components(self):
        return ["sbert-wk", "use"]

    def get_pca_size(self, concatenated_embedding):
        # Returns the number of components to use in PCA decomposition.
        return min(768, concatenated_embedding.shape[0]-1)
