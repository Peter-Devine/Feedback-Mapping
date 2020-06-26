from data.downloader import get_data

from mapping.mapper import map_data

from visualisation.visualise import visualise_data

from clustering.clusterer import cluster_data

from results.score_collator import collate_scores

from utils.utils import get_random_seed, set_random_seed

datasets = ["guzman_2015", "maalej_2016", "williams_2017", "chen_2014", "di_sorbo_2016", "scalabrino_2017", "jha_2017", "tizard_2019"]
embeddings = ["use",  "tfidf_pca", "glove", "bert_vanilla", "lda", "gem"]#"tfidf_pca", "glove", "sbert_wk", "ensemble"]

set_random_seed(111)

get_data(datasets)

map_data(datasets, embeddings)

visualise_data(datasets, embeddings)

cluster_data(datasets, embeddings)

collate_scores()
