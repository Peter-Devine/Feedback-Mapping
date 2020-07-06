from mapping.mapping_models.use import UseMapper
from mapping.mapping_models.bert_vanilla import BertVanillaMapper
from mapping.mapping_models.tfidf_pca import TfidfPcaMapper
from mapping.mapping_models.lda import LdaMapper
from mapping.mapping_models.glove import GloveMapper
from mapping.mapping_models.sbert import SBertMapper
from mapping.mapping_models.sbert_wk import SBertWKMapper
from mapping.mapping_models.gem import GemMapper
from mapping.mapping_models.t5_vanilla import T5VanillaMapper
from mapping.mapping_models.ensemble import EnsembleMapper
from mapping.mapping_models.trained_experiments.bert_cls_trained import BertClsTrainedMapper
from mapping.mapping_models.trained_experiments.bert_cls_trained_mtl import BertClsTrainedMtlMapper
from mapping.mapping_models.trained_experiments.bert_cls_trained_sim import BertClsTrainedSimMapper
from mapping.mapping_models.trained_experiments.bert_nsp_trained import BertNspTrainedMapper
from mapping.mapping_models.trained_experiments.bert_nsp_trained_mtl import BertNspTrainedMtlMapper
from mapping.mapping_models.trained_experiments.bert_nsp_trained_test import BertNspTrainedTestMapper
from mapping.mapping_models.trained_experiments.bert_nsp_trained_test_and_train import BertNspTrainedTestAndTrainMapper
from mapping.mapping_models.trained_experiments.bert_paired_trained import BertPairedTrainedMapper
from mapping.mapping_models.random import RandomMapper

MAPPER_DICT = {
    "use": UseMapper,
    "bert_vanilla": BertVanillaMapper,
    "tfidf_pca": TfidfPcaMapper,
    "lda": LdaMapper,
    "glove": GloveMapper,
    "sbert": SBertMapper,
    "sbert_wk": SBertWKMapper,
    "gem": GemMapper,
    "t5_vanilla": T5VanillaMapper,
    "ensemble": EnsembleMapper,
    "bert_cls_trained": BertClsTrainedMapper,
    "bert_cls_trained_mtl": BertClsTrainedMtlMapper,
    "bert_cls_trained_sim": BertClsTrainedSimMapper,
    "bert_nsp_trained": BertNspTrainedMapper,
    "bert_nsp_trained_test": BertNspTrainedTestMapper,
    "bert_nsp_trained_test_and_train": BertNspTrainedTestAndTrainMapper,
    "bert_nsp_trained_mtl": BertNspTrainedMtlMapper,
    "bert_paired_trained": BertPairedTrainedMapper,
    "random": RandomMapper,
}

def map_data(list_of_datasets, list_of_mappings):
    # If no mapping specified, run all
    if len(list_of_mappings) < 1:
        list_of_mappings = MAPPER_DICT.keys()

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
