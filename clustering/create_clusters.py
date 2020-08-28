import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from embed_evaluation.embed_evaluator import get_embeddings_df
from utils.utils import create_path

# Find all the embeddings, and cluster them
def cluster_embeds():
    # Get all available embeddings
    embed_dfs = get_embeddings_df()

    # Cycle through each dataset
    for dataset_name, dataset_dfs in embed_dfs.items():
        # Cycle through each app in a dataset
        for app_name, app_dfs in dataset_dfs.items():
            # Cycle through each embedding type in an app
            for embedding_name, (embedding_df, labels) in app_dfs.items():
                print(f"Evaluating embedding for {dataset_name} >> {app_name} >> {embedding_name}")
                # Get the score for an embedding
                cluster_embed(dataset_name, app_name, embedding_name, embedding_df, labels)

# Cluster the embeddings given
def cluster_embed(dataset_name, app_name, embedding_name, embedding_df, labels):
    clusterer = AgglomerativeClustering(n_clusters=len(labels.unique()))

    clusterer.fit(embedding_df.values)

    clustered_labels = clusterer.labels_

    cluster_labels_df = pd.DataFrame({"cluster_label": clustered_labels, "label":labels}, index=embedding_df.index)

    path = create_path([".", "data", "clusters", dataset_name, app_name, embedding_name])

    cluster_labels_df.to_csv(path)
