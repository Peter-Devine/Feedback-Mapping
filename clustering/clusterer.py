import os
import pandas as pd
import numpy as np
import scipy

from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score
from sklearn.metrics import pairwise_distances_argmin_min

from utils.utils import create_dir, get_random_seed

def cluster_data(list_of_datasets, list_of_embeddings):
    run_all = bool(len(list_of_embeddings) < 1)

    # Cycle through all the datasets provided
    for dataset_name in list_of_datasets:
        # Get all the embedding files available for this dataset
        embedding_files_data = get_embeddings_paths(dataset_name)

        for embedding_file, embedding_name in embedding_files_data:
            if run_all or embedding_name in list_of_embeddings:
                print(f"Clustering {dataset_name} in {embedding_name}")
                # Cluster single embedding from single dataset
                cluster_single_dataset(dataset_name, embedding_file, embedding_name)

def cluster_single_dataset(dataset_name, embedding_file, embedding_name):
    # Get embedding data
    embeddings, labels = get_embedding_data(embedding_file)

    # Cluster on embeddings and get predicted cluster membership
    preds, kmeans = cluster_kmeans(embeddings, labels)

    # Score predictions compared to golds
    scores = score_clustering(labels, preds, embeddings)

    # Get the compositions of each cluster
    compositions = get_cluster_compositions(labels, preds)

    # Get the indices of the observations closest to the centroid for each cluster
    closest_idx = get_examples_idx(kmeans, embeddings)

    # Get the text of the observations closest to the centroid for each cluster
    example_obs = get_examples_text(closest_idx, dataset_name)

    output_results(scores, compositions, example_obs, dataset_name, embedding_name)

def output_results(scores_df, compositions_df, example_obs_df, dataset_name, embedding_name):

    results_dir = os.path.join(".", "results")
    create_dir(results_dir)
    scores_dir = os.path.join(results_dir, "scores")
    create_dir(scores_dir)
    compositions_dir = os.path.join(results_dir, "compositions")
    create_dir(compositions_dir)
    examples_dir = os.path.join(results_dir, "examples")
    create_dir(examples_dir)
    dataset_scores_dir = os.path.join(scores_dir, dataset_name)
    create_dir(dataset_scores_dir)
    dataset_compositions_dir = os.path.join(compositions_dir, dataset_name)
    create_dir(dataset_compositions_dir)
    dataset_examples_dir = os.path.join(examples_dir, dataset_name)
    create_dir(dataset_examples_dir)

    scores_df.to_csv(os.path.join(dataset_scores_dir, f"{embedding_name}.csv"))
    compositions_df.to_csv(os.path.join(dataset_compositions_dir, f"{embedding_name}.csv"))
    example_obs_df.to_csv(os.path.join(dataset_examples_dir, f"{embedding_name}.csv"))

def get_embeddings_paths(dataset_name):
    # Get the directory where embeddings are saved, and make sure it is exists
    embedding_dir = os.path.join(".", "data", "embeddings", dataset_name)
    assert os.path.exists(embedding_dir), f"Cannot cluster as {embedding_dir} does not exist"

    # Lists all embedding files for a given dataset
    file_list = os.listdir(embedding_dir)

    # Get the file path and mapping name of each embedding file for the given dataset
    csv_file_list = []
    for file in file_list:
        if file[-4:] != ".csv":
            continue
        full_file_dir = os.path.join(embedding_dir, file)
        mapping_name = os.path.basename(file).replace(".csv", "")
        csv_file_list.append((full_file_dir, mapping_name))

    # Make sure we have at least one embedding file
    assert len(csv_file_list) > 0, f"Cannot cluster as there are {len(csv_file_list)} files in the embedding folder {embedding_dir}"

    return csv_file_list

def get_embedding_data(embedding_file):
    # Read embedding csv file to df
    embedding_df = pd.read_csv(embedding_file, index_col = 0)

    # Get gold label for each embedding
    labels = embedding_df["label"]

    # Get embedding values
    embeddings = embedding_df.drop("label", axis=1)

    return embeddings, labels

def cluster_kmeans(embeddings, labels):
    # Cluster values into k clusters where k is the number of unique labels in the dataset
    n_clusters = len(labels.unique())
    kmeans = KMeans(n_clusters=n_clusters, random_state=get_random_seed())

    # Get the predicted cluster number for each embedding
    preds = kmeans.fit_predict(embeddings)

    # Return predicted cluster numbers and cluster space object
    return preds, kmeans

def output_df(df, file_name):
    df.to_csv(f"{file_name}.csv")

def get_avg_max_fraction(labels, preds):
    max_fractions = []

    # Finds the maximum fraction for any one label in a fraction for each cluster, and then averages this value over all clusters
    for pred in np.unique(preds):
        in_cluster_labels = labels[preds==pred]

        max_fraction = in_cluster_labels.value_counts(normalize=True).max()

        max_fractions.append(max_fraction)

    if len(max_fractions) < 1:
        raise Exception(f"No clusters detected when comparing {labels} to {preds}. Max fractions were {max_fractions}")

    return sum(max_fractions) / len(max_fractions)

def get_knn_similarity_score(labels, embeddings, k=5):
    # Get the distances between each embedding
    mutual_distances = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(embeddings))

    # Find the arguments of the k closest points
    closest_args = [dist.argsort()[:k] for dist in mutual_distances]

    # Get a dict, mapping arguments to labels
    label_dict = {i: label for i, label in enumerate(labels)}

    # For each kth closest value, find the number of points which have the same label
    total_number_similar_neighbours = 0
    for kth_closest in range(k):
        closest_labels = np.vectorize(label_dict.get)(closest_args)[:,kth_closest]
        total_number_similar_neighbours += (closest_labels == labels).sum()

    # Get the average number of neighbours that have identical labels to a point
    avg_similar_neighbours = total_number_similar_neighbours / len(labels)

    return avg_similar_neighbours

def score_clustering(labels, preds, embeddings):
    # Score clustering predictions compared to real labels
    scorings = {}
    scorings["homogeneity"] = homogeneity_score(labels, preds)
    scorings["avg_max_frac"] = get_avg_max_fraction(labels, preds)
    scorings["num_clusters"] = len(np.unique(preds))
    scorings["knn_sim"] = get_knn_similarity_score(labels, embeddings, k=5)
    return pd.DataFrame(scorings, index=["scorings"])

def get_cluster_compositions(labels, preds):
    # Find what % of each cluster is any given label
    by_cluster_label_df = pd.DataFrame({"label": labels, "pred": preds}).groupby("pred")["label"]
    by_cluster_label_vc = by_cluster_label_df.value_counts()
    by_cluster_label_vc_norm = by_cluster_label_df.value_counts(normalize=True)
    return pd.DataFrame({"count": by_cluster_label_vc, "percentage": by_cluster_label_vc_norm})

def get_examples_idx(kmeans, embeddings):
    # Get the indices of the observations closest to the centroid for each cluster
    closests, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)

    # Since the pairwise_distances_argmin_min function returns positional indices, we map them back to df indices through the embeddings_df index
    closest_idx = embeddings.index[closests]
    return closest_idx

def get_examples_text(closest_idx, dataset_name):
    # Get the original text data of the dataset
    text_dir = os.path.join(".", "data", "raw", dataset_name, f"test.csv")
    text_df = pd.read_csv(text_dir, index_col = 0)

    # Get the observations that are closest to the centroid of each cluster
    return text_df.loc[closest_idx]
