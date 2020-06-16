from data.downloader import get_data

from mapping.mapper import map_data

from visualisation.visualise import visualise_data

from clustering.clusterer import cluster_data

datasets = ["williams_2017"]
embeddings = ["glove"]

get_data(datasets, random_state=111)

map_data(datasets, embeddings)

visualise_data(datasets, embeddings)

cluster_data(datasets, embeddings)
