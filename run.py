from data.downloader import get_data

from mapping.mapper import map_data

from visualisation.visualise import visualise_data

from clustering.clusterer import cluster_data

get_data(["guzman_2015"], random_state=111)

map_data(["guzman_2015"], ["tfidf_pca"])

visualise_data(["guzman_2015"])

cluster_data(["guzman_2015"])
