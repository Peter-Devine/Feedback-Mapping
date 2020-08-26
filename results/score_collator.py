import os
import pandas as pd
import matplotlib.pyplot as plt

def collate_scores(metric):
    print("Creating collated score csv file")
    scores_path = os.path.join(".", "results", "scores")

    collated_scores = {}

    for dataset_name in os.listdir(scores_path):
        dataset_score_path = os.path.join(scores_path, dataset_name)
        # If the item in the scores folder is a folder, open it and collect the contents of it
        if os.path.isdir(dataset_score_path):

            # Go through all apps in a dataset
            for app_name in os.listdir(dataset_score_path):
                app_score_path = os.path.join(dataset_score_path, app_name)

                app_dataset_name = f"{dataset_name[:3]}_{app_name[:8]}"

                collated_scores[app_dataset_name] = {}

                # Iterate through all score files for every embedding
                for score_file in os.listdir(app_score_path):
                    # Check that the score file is a csv file
                    if score_file[-4:] == ".csv":
                        score_file_path = os.path.join(app_score_path, score_file)

                        # Read the contents of the score file
                        print(f"Reading score of {dataset_name} >> {app_name} >> {score_file}")
                        score_df = pd.read_csv(score_file_path, index_col=0)

                        score = score_df.loc["scorings", metric]

                        embedding_name = score_file[:-4]
                        collated_scores[app_dataset_name][embedding_name] = score

    collated_scores_path = os.path.join(scores_path, f"{metric}_collated_score.csv")

    collated_df = pd.DataFrame(collated_scores).T

    collated_df.to_csv(collated_scores_path)

    indices = collated_df.mean().sort_values(ascending=True).index
    collated_df.loc[:, indices].boxplot(figsize=(11,6))
    ax = collated_df.loc[:, indices].plot.box(figsize=(30,18))#.legend(loc='center left',bbox_to_anchor=(1.0, 0.5))
    ax.set_xticklabels([i.replace("_", "\n") for i in indices])

    plt.savefig(os.path.join(scores_path, f"{metric}_collated_boxplot.png"))

    plt.close()

    ax = collated_df.rank(ascending = False, axis=1).mean().sort_values().plot.bar(figsize=(15,9))
    ax.set_ylabel("Average ranking")

    plt.savefig(os.path.join(scores_path, f"{metric}_average_ranking.png"))

    plt.close()


    ax = collated_df.plot.bar(figsize=(16,9))
    ax.set_xticklabels([i.replace("_", "\n") for i in collated_df.index])

    plt.savefig(os.path.join(scores_path, f"{metric}_collated_barchart.png"))

    plt.close()
