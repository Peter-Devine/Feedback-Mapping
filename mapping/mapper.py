###################################################################

######################################################
################# Data-blind models ##################
######################################################

### OOTB embedders (BERT-like)
from mapping.mapping_models.data_blind_models.bert_likes.albert_vanilla import AlbertVanillaMapper
from mapping.mapping_models.data_blind_models.bert_likes.bart_vanilla import BartVanillaMapper
from mapping.mapping_models.data_blind_models.bert_likes.bert_vanilla_first_tok import BertVanillaFirstTokMapper
from mapping.mapping_models.data_blind_models.bert_likes.bert_vanilla_large import BertVanillaLargeMapper
from mapping.mapping_models.data_blind_models.bert_likes.bert_vanilla import BertVanillaMapper
from mapping.mapping_models.data_blind_models.bert_likes.distilbert_vanilla import DistilbertVanillaMapper
from mapping.mapping_models.data_blind_models.bert_likes.roberta_vanilla_large import RobertaVanillaLargeMapper

### Outside trained in-domain data blind models
from mapping.mapping_models.data_blind_models.outside_trained.bert_cls_trained_sublabel_outside import BertClsTrainedSublabelOutsideMapper
from mapping.mapping_models.data_blind_models.outside_trained.bert_masking_trained_outside import BertMaskingTrainedOutsideMapper
from mapping.mapping_models.data_blind_models.outside_trained.bert_nsp_cos_trained_outside import BertNspCosTrainedOutsideMapper
from mapping.mapping_models.data_blind_models.outside_trained.bert_nsp_trained_outside import BertNspTrainedOutsideMapper

### Other OOTB trained models
from mapping.mapping_models.data_blind_models.gem import GemMapper
from mapping.mapping_models.data_blind_models.glove import GloveMapper
from mapping.mapping_models.data_blind_models.keyword_matching import KeywordMatchingMapper
from mapping.mapping_models.data_blind_models.random import RandomMapper
from mapping.mapping_models.data_blind_models.sbert_wk import SBertWKMapper
from mapping.mapping_models.data_blind_models.sbert import SBertMapper
from mapping.mapping_models.data_blind_models.t5_vanilla import T5VanillaMapper
from mapping.mapping_models.data_blind_models.use import UseMapper

##########################################################
################### Data-fit models ######################
##########################################################

### Classical models
from mapping.mapping_models.data_fit_models.classical_models.gsdmm import GsdmmMapper
from mapping.mapping_models.data_fit_models.classical_models.gsdmm_small import GsdmmSmallMapper
from mapping.mapping_models.data_fit_models.classical_models.lda import LdaMapper
from mapping.mapping_models.data_fit_models.classical_models.lda_small import LdaSmallMapper
from mapping.mapping_models.data_fit_models.classical_models.tfidf_pca import TfidfPcaMapper
from mapping.mapping_models.data_fit_models.classical_models.tfidf_pca_small import TfidfPcaSmallMapper

### Label trained language models
from mapping.mapping_models.data_fit_models.label_lm.bert_cls_trained_mtl_except import BertClsTrainedMtlExceptMapper

### Masking trained language models
from mapping.mapping_models.data_fit_models.masking_lm.bert_masking_trained_app import BertMaskingTrainedAppMapper
from mapping.mapping_models.data_fit_models.masking_lm.bert_masking_trained_dataset import BertMaskingTrainedDatasetMapper
from mapping.mapping_models.data_fit_models.masking_lm.bert_masking_trained_mtl import BertMaskingTrainedMtlMapper

### Metadata trained language models
from mapping.mapping_models.data_fit_models.metadata_lm.bert_cls_trained_sublabel_app import BertClsTrainedSublabelAppMapper
from mapping.mapping_models.data_fit_models.metadata_lm.bert_cls_trained_sublabel_dataset import BertClsTrainedSublabelDatasetMapper
from mapping.mapping_models.data_fit_models.metadata_lm.bert_cls_trained_sublabel_mtl import BertClsTrainedSublabelMtlMapper

### Next sentence prediction trained language models
from mapping.mapping_models.data_fit_models.nsp_lm.bert_nsp_cos_trained_app import BertNspCosTrainedAppMapper
from mapping.mapping_models.data_fit_models.nsp_lm.bert_nsp_cos_trained_dataset import BertNspCosTrainedDatasetMapper
from mapping.mapping_models.data_fit_models.nsp_lm.bert_nsp_cos_trained_mtl import BertNspCosTrainedMtlMapper
from mapping.mapping_models.data_fit_models.nsp_lm.bert_nsp_trained_app import BertNspTrainedAppMapper
from mapping.mapping_models.data_fit_models.nsp_lm.bert_nsp_trained_dataset import BertNspTrainedDatasetMapper
from mapping.mapping_models.data_fit_models.nsp_lm.bert_nsp_trained_mtl import BertNspTrainedMtlMapper

MAPPER_DICT = {
    # Data fit classical models
    "albert_vanilla": AlbertVanillaMapper,
    "bart_vanilla": BartVanillaMapper,
    "bert_vanilla_first_tok": BertVanillaFirstTokMapper,
    "bert_vanilla_large": BertVanillaLargeMapper,
    "bert_vanilla": BertVanillaMapper,
    "distilbert_vanilla": DistilbertVanillaMapper,
    "roberta_vanilla_large": RobertaVanillaLargeMapper,

    "bert_cls_trained_sublabel_outside": BertClsTrainedSublabelOutsideMapper,
    "bert_masking_trained_outside": BertMaskingTrainedOutsideMapper,
    "bert_nsp_cos_trained_outside": BertNspCosTrainedOutsideMapper,
    "bert_nsp_trained_outside": BertNspTrainedOutsideMapper,

    "gem": GemMapper,
    "glove": GloveMapper,
    "keyword_matching": KeywordMatchingMapper,
    "random": RandomMapper,
    "sbert_wk": SBertWKMapper,
    "sbert": SBertMapper,
    "t5_vanilla": T5VanillaMapper,
    "use": UseMapper,

    "gsdmm": GsdmmMapper,
    "gsdmm_small": GsdmmMapper,
    "lda": LdaMapper,
    "lda_small": LdaSmallMapper,
    "tfidf_pca": TfidfPcaMapper,
    "tfidf_pca_small": TfidfPcaSmallMapper,

    "bert_cls_trained_mtl_except": BertClsTrainedMtlExceptMapper,

    "bert_masking_trained_app": BertMaskingTrainedAppMapper,
    "bert_masking_trained_dataset": BertMaskingTrainedDatasetMapper,
    "bert_masking_trained_mtl": BertMaskingTrainedMtlMapper,

    "bert_cls_trained_sublabel_app": BertClsTrainedSublabelAppMapper,
    "bert_cls_trained_sublabel_dataset": BertClsTrainedSublabelDatasetMapper,
    "bert_cls_trained_sublabel_mtl": BertClsTrainedSublabelMtlMapper,

    "bert_nsp_cos_trained_app": BertNspCosTrainedAppMapper,
    "bert_nsp_cos_trained_dataset": BertNspCosTrainedDatasetMapper,
    "bert_nsp_cos_trained_mtl": BertNspCosTrainedMtlMapper,
    "bert_nsp_trained_app": BertNspTrainedAppMapper,
    "bert_nsp_trained_dataset": BertNspTrainedDatasetMapper,
    "bert_nsp_trained_mtl": BertNspTrainedMtlMapper,
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
