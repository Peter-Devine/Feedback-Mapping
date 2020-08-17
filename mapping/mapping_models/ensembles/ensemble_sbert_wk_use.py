from mapping.mapping_models.ensembles.ensemble_base import EnsembleBaseMapper

class EnsembleSbertWkUseMapper(EnsembleBaseMapper):

    def get_ensemble_components(self):
        return ["sbert-wk", "use"]
