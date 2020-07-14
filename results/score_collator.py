import os
import pandas as pd
import matplotlib.pyplot as plt

def collate_scores():
    print("Creating collated score csv file")
    scores_path = os.path.join(".", "results", "scores")

    collated_scores = {}

    for item in os.listdir(scores_path):
        dataset_score_path = os.path.join(scores_path, item)
        # If the item in the scores folder is a folder, open it and collect the contents of it
        if os.path.isdir(dataset_score_path):

            collated_scores[item] = {}

            # Iterate through all score files for every embedding
            for score_file in os.listdir(dataset_score_path):
                # Check that the score file is a csv file
                if score_file[-4:] == ".csv":
                    score_file_path = os.path.join(dataset_score_path, score_file)

                    # Read the contents of the score file
                    score_df = pd.read_csv(score_file_path, index_col=0)

                    score = score_df.loc["scorings", "homogeneity"]

                    embedding_name = score_file[:-4]
                    collated_scores[item][embedding_name] = score

    collated_scores_path = os.path.join(scores_path, "collated_score.csv")

    collated_df = pd.DataFrame(collated_scores).T

    collated_df.to_csv(collated_scores_path)

    collated_df.loc[:, collated_df.max().sort_values(ascending=True).index].boxplot(figsize=(11,6))

    indices = collated_df.max().sort_values(ascending=True).index
    ax = collated_df.loc[:, indices].boxplot(figsize=(16,9))#.legend(loc='center left',bbox_to_anchor=(1.0, 0.5))
    ax.set_xticklabels([i.replace("_", "\n") for i in indices])

    plt.savefig(os.path.join(scores_path, "collated_boxplot.png"))

    plt.close()

    ax = collated_df.plot.bar(figsize=(16,9))
    ax.set_xticklabels([i.replace("_", "\n") for i in collated_df.index])

    plt.savefig(os.path.join(scores_path, "collated_barchart.png"))

    plt.close()
