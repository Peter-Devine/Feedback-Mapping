from mapping.mapping_models.use import UseMapper
from mapping.mapping_models.bert_vanilla import BertVanillaMapper
from mapping.mapping_models.tfidf_pca import TfidfPcaMapper
from mapping.mapping_models.glove import GloveMapper
from mapping.mapping_models.sbert import SBertMapper
from mapping.mapping_models.sbert_wk import SBertWKMapper

MAPPER_DICT = {
    "use": UseMapper,
    "bert_vanilla": BertVanillaMapper,
    "tfidf_pca": TfidfPcaMapper,
    "glove": GloveMapper,
    "sbert": SBertMapper,
    "sbert_wk": SBertWKMapper,
}

def map_data(list_of_datasets, list_of_mappings):
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
