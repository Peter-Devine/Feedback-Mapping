import os
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score
from sklearn.metrics import pairwise_distances_argmin_min

def cluster_data(list_of_datasets):
    # Cycle through all the datasets provided
    for dataset in list_of_datasets:
        # Get all the embedding files available for this dataset
        embedding_files = get_embeddings_paths(dataset)
        for embedding_file in embedding_files:
            cluster_given_file()

def get_embeddings_paths(dataset_name):
    # Get the directory where embeddings are saved, and make sure it is exists
    embedding_dir = os.path.join(".", "data", "embeddings", dataset_name)
    assert os.path.exists(embedding_dir), f"Cannot cluster as {embedding_dir} does not exist"

    # Get the files in this directory, and out of those files, filter out non-csv files
    file_list = os.listdir(embedding_dir)
    csv_file_list = [x for x in file_list if x[-4:] == ".csv"]

    # Make sure we have at least one embedding file
    assert len(csv_file_list) > 0, f"Cannot cluster as there are {len(csv_file_list)} files in the embedding folder {embedding_dir}"

    return csv_file_list

def cluster_given_file(embedding_file):
    df = pd.read_csv(embedding_file)

    kmeans = KMeans(n_clusters=len(df.label.unique()), random_state=0)
    preds = kmeans.fit_predict(df.drop("label", axis=1))
    score = homogeneity_score(df.label, preds)

    closests, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, df.drop("label", axis=1))
