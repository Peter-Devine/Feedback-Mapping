import os
import pandas as pd

# Get the closest embedding for each label-dataset to compare proficiency between datasets
# E.g. we find, for each reference point (one per label per app-dataset), the closest review/tweet/forum post/ etc.
# that has been embedded to said reference point.
def collate_examples():
    print("Creating collated examples csv file")
    examples_path = os.path.join(".", "results", "examples")

    collated_examples = {}

    for dataset_name in os.listdir(examples_path):
        dataset_example_path = os.path.join(examples_path, dataset_name)
        # If the item in the scores folder is a folder, open it and collect the contents of it
        if os.path.isdir(dataset_example_path):

            # Go through all apps in a dataset
            for app_name in os.listdir(dataset_example_path):
                app_example_path = os.path.join(dataset_example_path, app_name)

                app_dataset_name = f"{dataset_name[:3]}_{app_name[:8]}"

                collated_examples[app_dataset_name] = {}

                # Iterate through all example files for every embedding
                for example_file in os.listdir(app_example_path):
                    # Check that the example file is a csv file
                    if example_file[-4:] == ".csv":
                        example_file_path = os.path.join(app_example_path, example_file)

                        # Read the contents of the example file
                        print(f"Reading examples of {dataset_name} >> {app_name} >> {example_file}")
                        example_df = pd.read_csv(score_file_path, index_col=0)

                        # Get the reference text and the single closest text to that reference text
                        ref_example = example_df["text"].iloc[0]
                        best_example = example_df["text"].iloc[1]

                        # The name of the example .csv files is "[EMBEDDING NAME]__[LABEL NAME]", so we split them and get the individually
                        embedding_label_name = example_file[:-4]
                        embedding_name = embedding_label_name.split("__")[0]
                        label_name = "__".join(embedding_label_name.split("__")[1:])
                        collated_examples[f"{app_dataset_name}_{label_name}"]["Reference"] = ref_example
                        collated_examples[f"{app_dataset_name}_{label_name}"][embedding_name] = best_example

    collated_examples_path = os.path.join(examples_path, "collated_examples.csv")

    collated_df = pd.DataFrame(collated_examples)

    collated_df.to_csv(collated_examples_path)
