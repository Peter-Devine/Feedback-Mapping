from data.downloader import get_data

from mapping.mapper import map_data

from visualisation.visualise import visualise_data

from clustering.clusterer import cluster_data

from utils.utils import get_random_seed, set_random_seed

datasets = ["guzman_2015", "williams_2017"]
embeddings = ["bert_cls_trained"]

set_random_seed(111)

get_data(datasets)

map_data(datasets, embeddings)

visualise_data(datasets, embeddings)

cluster_data(datasets, embeddings)
