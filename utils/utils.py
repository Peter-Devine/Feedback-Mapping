import os
import random

def create_dir(dir):
    if not os.path.exists(dir):
        print(f"Creating output directory at {dir}")
        os.mkdir(dir)

def create_path(list_of_dirs):
    all_dir = ""
    for dir in list_of_dirs:
        all_dir = os.path.join(all_dir, dir)
        create_dir(all_dir)
    return all_dir

seed_env_var_name = "PYTHON_RANDOM_SEED_CLUSTERING"

def get_random_seed():
    try:
        seed_value_str = os.environ[seed_env_var_name]
        seed_value = int(seed_value_str)
        return seed_value
    except Exception as err:
        raise Exception("Random seed has not been set yet. Please set before running.")

def set_random_seed(seed_value):
    os.environ[seed_env_var_name] = str(seed_value)

def randomly_shuffle_list(input_list):
    random.seed(a=get_random_seed())
    random.shuffle(input_list)

def randomly_sample_list(input_list, k=1):
    random.seed(a=get_random_seed())
    return random.sample(input_list, k=1)

def split_df(df, split_frac=0.7, has_labels=True):
    train_df = df.sample(frac=split_frac, random_state=get_random_seed())
    valid_df = df.drop(train_df.index)

    if has_labels:
        # Make sure that the validation set does not have labels that were not included in train
        valid_df_labels = train_df.label.unique()
        valid_df = valid_df[valid_df.label.isin(valid_df_labels)]

    return train_df, valid_df

# Takes a dict of dfs ({"A": df1, "B": df2}), and appends them like df1.append(df2)
def combine_dict_of_dfs_text(dict_of_dfs):
    combined_df = None
    for key, df in dict_of_dfs.items():
        if combined_df is None:
            combined_df = df["text"]
        else:
            combined_df = combined_df.append(df["text"]).reset_index(drop=True)
    return combined_df

# Takes a dict of dicts of dfs {"Task1": {"AppA": df1, "AppB": df2}}, and returns a combined df of their texts
def get_all_dataset_combined_text(all_dataset_dict):
    combined_df = None

    for dataset_name, dataset_dict in all_dataset_dict.items():
        dataset_df = combine_dict_of_dfs_text(dataset_dict)

        if combined_df is None:
            combined_df = dataset_df
        else:
            combined_df = combined_df.append(dataset_df).reset_index(drop=True)

    return combined_df

def bad_char_del(text):
     return text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
