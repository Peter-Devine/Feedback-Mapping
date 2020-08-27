from mapping.mapping_models.ensembles.ensemble_base import EnsembleBaseMapper

class EnsembleSbertWkUsePcaMapper(EnsembleBaseMapper):

    def get_ensemble_components(self):
        return ["sbert-wk", "use"]

    def get_pca_size(self):
        # Returns the number of components to use in PCA decomposition.
        return 768
