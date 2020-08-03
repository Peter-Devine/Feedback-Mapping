from mapping.mapping_models.data_fit_models.classical_models.lda import LdaMapper
from mapping.mapping_models.data_fit_models.classical_models.lda_small import LdaSmallMapper
from mapping.mapping_models.data_fit_models.classical_models.tfidf_pca import TfidfPcaMapper
from mapping.mapping_models.data_fit_models.classical_models.tfidf_pca_small import TfidfPcaSmallMapper

MAPPER_DICT = {
    # Data fit classical models
    "tfidf_pca": TfidfPcaMapper,
    "tfidf_pca_small": TfidfPcaSmallMapper,
    "lda": LdaMapper,
    "lda_small": LdaSmallMapper,
}

def map_data(list_of_datasets, list_of_mappings):
    # If no mapping specified, run all
    if len(list_of_mappings) < 1:
        list_of_mappings = MAPPER_DICT.keys()

        # Remove the mappings that cannot be run on most datasets / don't play well with other mappings (USE and TF2.0 take all the memory away from PyTorch)
        list_of_mappings.remove("use")

    for dataset in list_of_datasets:
        for mapping in list_of_mappings:
            embed_single_dataset(mapping, dataset)

def embed_single_dataset(mapping, dataset):
    mapper = get_mapper(mapping, dataset)
    mapper.embed()

def get_mapper(mapping_type, dataset):
    mapping_type = mapping_type.lower()
    assert mapping_type in MAPPER_DICT.keys(), f"{mapping_type} mapping type unsupported.\nSupported mappings are {MAPPER_DICT.keys()}."
    mapper_class = MAPPER_DICT[mapping_type]
    mapper = mapper_class(test_dataset=dataset)
    return mapper
