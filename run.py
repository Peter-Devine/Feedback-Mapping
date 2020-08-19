from data.downloader import get_data

from mapping.mapper import map_data

from embed_evaluation.embed_evaluator import eval_embeds

from results.score_collator import collate_scores

from utils.utils import set_random_seed

datasets = ["guzman_2015", "maalej_2016", "williams_2017", "chen_2014", "di_sorbo_2016", "scalabrino_2017", "jha_2017", "tizard_2019"]
embeddings = [
 #'albert_vanilla',
 'bart_vanilla',
 'bert_vanilla_first_tok',
 'bert_vanilla_large',
 'bert_vanilla',
 'distilbert_vanilla',
 'declutr_vanilla',
 'roberta_vanilla_large',
 'sentence_transformer_bert',
 'sentence_transformer_distilbert',
 'sentence_transformer_roberta',
 'bert_cls_trained_sublabel_outside',
 'bert_masking_trained_outside',
 'bert_nsp_cos_trained_outside',
 'bert_nsp_trained_outside',
 'gem',
 'glove',
 'keyword_matching',
 'random',
 'sbert_wk',
 'sbert',
 't5_vanilla',
 #'use',
 'gsdmm',
 'gsdmm_small',
 'lda',
 'lda_small',
 'tfidf_pca',
 'tfidf_pca_small',
 'bert_cls_trained_mtl_except',
 'bert_masking_trained_app',
 #'bert_masking_trained_dataset',
 'bert_masking_trained_mtl',
 'bert_cls_trained_sublabel_app',
 #'bert_cls_trained_sublabel_dataset',
 'bert_cls_trained_sublabel_mtl',
 'bert_nsp_cos_trained_app',
 #'bert_nsp_cos_trained_dataset',
 'bert_nsp_cos_trained_mtl',
 'bert_nsp_trained_app',
 #'bert_nsp_trained_dataset',
 'bert_nsp_trained_mtl']

set_random_seed(111)

get_data(datasets)

map_data(datasets, embeddings)

eval_embeds()

collate_scores("5_nn_sim")

collate_scores("5_cos_nn_sim")

collate_scores("p_val")
