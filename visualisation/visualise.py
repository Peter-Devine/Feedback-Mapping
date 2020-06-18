import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pandas as pd

from sklearn.decomposition import PCA

from clustering.clusterer import get_embeddings_paths, get_embedding_data
from utils.utils import create_dir

def visualise_data(dataset_names, list_of_embeddings):

    run_all = bool(len(list_of_embeddings) < 1)

    for dataset_name in dataset_names:
        # Get all embedding files for given dataset
        embedding_files_data = get_embeddings_paths(dataset_name)

        for embedding_file, embedding_name in embedding_files_data:
            if run_all or embedding_name in list_of_embeddings:
                print(f"Visualising {dataset_name} in {embedding_name}")
                # Create the visulisation for each embedding and dataset
                visualise_single_dataset(dataset_name, embedding_file, embedding_name)

def visualise_single_dataset(dataset_name, embedding_file, embedding_name):
    # From each embedding file, takeout the numerical embeddings and labels
    embeddings, labels = get_embedding_data(embedding_file)
    # Map the high dim embeddings to a 2D space
    twoD_mapping_df = get_PCA_two_D_df(embeddings, labels)
    # Output the resulting 2D space to a graph
    graph_PCA_data(twoD_mapping_df, dataset_name, embedding_name)

def get_PCA_two_D_df(embeddings, labels):
    # Initialize the PCA model to map multi dimensional embeddings to simple 2D embedding
    pca = PCA(n_components=2)

    # Fit this model to the given data and transform it to 2D space
    twoD_mapping = pca.fit_transform(embeddings)

    # Make the embeddings into a DataFrame
    twoD_mapping_df = pd.DataFrame(twoD_mapping, index=embeddings.index)

    # Add the labels of each embedding to the dataframe
    twoD_mapping_df["label"] = labels

    return twoD_mapping_df

def get_colour_map(unique_labels):
    # Set some colours that contrast nicely with eachother
    set_colours = ["k", "r", "y", "b", "m", "g", "c", "orange", "silver", "purple", "greenyellow", "cadetblue", "sienna"]
    # Get the number of unique labels in the dataset
    num_labels = len(unique_labels)
    # Get the RGB values of colours moving successively along the HSV spectrum
    colours = cm.hsv(np.linspace(0, 1, num_labels))
    # Convert RGB values to hex strings
    colours = ["#%02x%02x%02x" % (int(r*255), int(g*255), int(b*255)) for r,g,b,a in colours]
    # add the set colours to the hex colours
    if num_labels <= len(set_colours):
        colours = set_colours[:num_labels]
    else:
        extra_colours = num_labels - len(set_colours)
        colours = set_colours + colours[:extra_colours]
    # Map each hex string to a certain unique label
    colour_map = {label:colour for label, colour in zip(unique_labels, colours)}
    return colour_map

def get_graph_name(dataset_name, mapping_name):
    results_dir = os.path.join(".", "results")
    create_dir(results_dir)
    visualisations_dir = os.path.join(results_dir, "visualisations")
    create_dir(visualisations_dir)
    dataset_results_dir = os.path.join(visualisations_dir, dataset_name)
    create_dir(dataset_results_dir)
    mapping_results_file = os.path.join(dataset_results_dir, f"{mapping_name}.png")
    return mapping_results_file

def graph_PCA_data(twoD_mapping_df, dataset_name, mapping_name):
    unique_labels = twoD_mapping_df["label"].unique()

    colour_map = get_colour_map(unique_labels)

    plt.figure(figsize=(11, 8))

    for idx, row in twoD_mapping_df.groupby("label"):
        plt.scatter(x=row[0], y=row[1], c=colour_map[idx], label=idx)

    plt.legend()

    file_name = get_graph_name(dataset_name, mapping_name)

    plt.savefig(file_name)

    plt.close()
