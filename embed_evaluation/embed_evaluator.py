import os
import pandas as pd
import numpy as np
import scipy

from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score
from sklearn.metrics import pairwise_distances_argmin_min

from utils.utils import create_dir, get_random_seed

# Find all the embeddings, and evaluate them
def eval_embeds():
    # Get all available embeddings
    embed_dfs = get_embeddings_df()

    # Cycle through each dataset
    for dataset_name, dataset_dfs in embed_dfs.items():
        # Cycle through each app in a dataset
        for app_name, app_dfs in dataset_dfs.items():
            # Cycle through each embedding type in an app
            for embedding_name, (embedding_df, labels) in app_dfs.items():
                # Get the score for an embedding
                eval_embed(dataset_name, app_name, embedding_name, embedding_df, labels)

# Get scorings etc. on a given embedding
def eval_embed(dataset_name, app_name, embedding_name, embedding_df, labels):
    embeddings = embedding_df.values

    # Get basic info on labels (number of each label etc.)
    basic_label_info(dataset_name, app_name, embedding_name, labels)

    # Score the embeddings quantitavely
    score_embeddings(dataset_name, app_name, embedding_name, embeddings, labels)

    # Get some qualitative measures to intuitively evaluate the embeddings (sanity check)
    get_k_closest_example_points(dataset_name, app_name, embedding_name, embeddings, labels, k=20)

# As a sanity check as much as anything, we look at what the distribution of labels are etc.
def basic_label_info(dataset_name, app_name, embedding_name, labels):
    # Find both the gross and relative abundance of labels
    vc = labels.value_counts(normalize=True)
    vc_norm = labels.value_counts(normalize=False)
    label_info_df = pd.DataFrame({"counts": vc, "norm" : vc_norm})
    output_results_df("label_info", dataset_name, app_name, embedding_name, label_info_df)

# Score the given embedding relative to multiple metrics
def score_embeddings(dataset_name, app_name, embedding_name, embedding_df, labels):
    embed_scores = {}
    embed_scores["5_nn_sim"] = get_knn_similarity_score(embedding_df, labels, k=5)
    embed_scores["10_nn_sim"] = get_knn_similarity_score(embedding_df, labels, k=10)
    embed_scores["15_nn_sim"] = get_knn_similarity_score(embedding_df, labels, k=15)
    embed_scores["5_cos_nn_sim"] = get_knn_similarity_score(embedding_df, labels, k=5, metric="cosine")
    embed_scores["10_cos_nn_sim"] = get_knn_similarity_score(embedding_df, labels, k=10, metric="cosine")
    embed_scores["15_cos_nn_sim"] = get_knn_similarity_score(embedding_df, labels, k=15, metric="cosine")

    # Get this scoring into a df form
    df = pd.DataFrame(scorings, index=["scorings"])

    # Output this df to disk
    output_results_df("scores", dataset_name, app_name, embedding_name, df)

# Saves the given df to ./results/[dataset_name]/[app_name]/[embedding_name].csv
def output_results_df(result_type, dataset_name, app_name, embedding_name, df):
    app_dir = create_results_dirs(result_type, dataset_name, app_name)
    file_path = os.path.join(app_dir, embedding_name)
    output_df(df, file_path)

# Makes sure that the ./results/[dataset_name]/[app_name] dir exists
def create_results_dirs(result_type, dataset_name, app_name):
    all_results_dir = os.path.join(".", "results")
    create_dir(all_results_dir)
    results_dir = os.path.join(all_results_dir, result_type)
    create_dir(results_dir)
    dataset_dir = os.path.join(results_dir, dataset_name)
    create_dir(dataset_dir)
    app_dir = os.path.join(dataset_dir, app_name)
    create_dir(app_dir)
    return app_dir

# Gets a dict of dfs for each embedding in each app in each dataset.
# Layout of this dict is:
# {
#     "dataset1": {
#         "app1": {
#             "embedding1": embedding_df
#         }
#     }
# }
def get_embeddings_df():
    # Get the directory where embeddings are saved, and make sure it is exists
    embedding_dir = os.path.join(".", "data", "embeddings")
    assert os.path.exists(embedding_dir), f"Cannot cluster as {embedding_dir} does not exist"

    embeddings_dfs = {}

    # Cycle through all the datasets
    for dataset_folder in os.listdir(embedding_dir):
        embeddings_dfs[dataset_folder] = {}
        dataset_folder_dir =  os.path.join(embedding_dir, dataset_folder)

        # Cycle through all the apps in the dataset
        for app_folder in os.listdir(dataset_folder_dir):
            embeddings_dfs[dataset_folder][app_folder] = {}
            app_folder_dir =  os.path.join(dataset_folder_dir, app_folder)

            # Cycle through all the embeddings an app
            for embed_file in os.listdir(app_folder_dir):
                embed_file_dir =  os.path.join(app_folder_dir, embed_file)

                embed_file_name = embed_file.replace(".csv", "")
                embeddings_dfs[dataset_folder][app_folder][embed_file_name] = get_embedding_data(embed_file_dir)

    return embeddings_dfs

# Gets the embeddings and labels fr an embedding given the file path
def get_embedding_data(embedding_file):
    # Read embedding csv file to df
    embedding_df = pd.read_csv(embedding_file, index_col = 0)

    # Get gold label for each embedding (as fine as possible)
    labels = embedding_df["label"]

    # Get embedding values
    embeddings = embedding_df.drop("label", axis=1)

    return embeddings, labels

def output_df(df, file_name):
    df.to_csv(f"{file_name}.csv")

# Find the distance between each embedding
def get_pairwise_dist(embeddings, metric):
    return scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(embeddings, metric=metric))

# Finds the average number of points with the same label in the k nearest points to any one point.
def get_knn_similarity_score(embeddings, labels, k=5, metric="euclidean"):
    # Get the distances between each embedding
    mutual_distances = get_pairwise_dist(embeddings, metric=metric)
    print(mutual_distances.shape)

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

# Gets the text of example points
def get_k_closest_example_points(dataset_name, app_name, embedding_name, embeddings, labels, k=20):
    # First, find the euclidean distances between every embedding
    mutual_distances = get_pairwise_dist(embeddings, metric="euclidean")
    closest_args = dist.argsort()[:, :k]

    # We also find the cos distance to see if there are any great discrepancies between cos and euclidean
    mutual_cos_distances = get_pairwise_dist(embeddings, metric="cosine")

    all_points_df = None

    for label in labels.unique():
        # Get the randomly sampled reference points position
        ref_idx = labels[labels == label].sample(n = 1, random_state = get_random_seed()).index.iloc[0]
        ref_numerical_idx = labels.index.get_loc(ref_idx)

        # Find the numerical position and distance of the k nearest points to the reference point
        nearest_points_numerical_idx = closest_args[ref_numerical_idx]
        nearest_points_distances = mutual_distances[ref_numerical_idx, nearest_points_numerical_idx]
        nearest_points_cos_distances = mutual_cos_distances[ref_numerical_idx, nearest_points_numerical_idx]

        # Get the indices of reference point + knn
        select_idx = [ref_numerical_idx] + [x for x in nearest_points_numerical_idx]

        # Get the accompanying text and labels of these points
        nearest_df = get_examples_text(nearest_points_numerical_idx, dataset_name, app_name)
        nearest_df["point_num"] = [f"{label}_point_{i}" for i in range(nearest_df.shape[0])]
        nearest_df["euclidean_distance"] = [0] + [x for x in nearest_points_distances]
        nearest_df["cosine_distance"] = [0] + [x for x in nearest_points_cos_distances]

        nearest_df = nearest_df.set_index("point_num", drop=False)

        if all_points_df is None:
            all_points_df = nearest_df
        else:
            all_points_df.append(nearest_df)

    output_results_df("examples", dataset_name, app_name, embedding_name, all_points_df)

def get_examples_text(idx, dataset_name, app_name):
    # Get the original text data of the dataset
    text_dir = os.path.join(".", "data", "raw", dataset_name, f"{app_name}.csv")
    text_df = pd.read_csv(text_dir, index_col = 0)

    # Get the observations from the dataset that correspond to the supplied idx
    selected_df = text_df.iloc[idx]
    # Only output the text and label of the found points (the output looks cluttered otherwise)
    selected_df = selected_df[["text", "label"]]

    # Return the text and labels from the selected observations of this dataset
    return selected_df
