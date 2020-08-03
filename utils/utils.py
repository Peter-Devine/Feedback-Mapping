import os
import random

def create_dir(dir):
    if not os.path.exists(dir):
        print(f"Creating output directory at {dir}")
        os.mkdir(dir)

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

def randomly_shuffle_list(list):
    random.seed(a=get_random_seed())
    random.shuffle(list)

def split_df(df, split_frac=0.7):
    train_df = df.sample(frac=split_frac, random_state=get_random_seed())
    valid_df = df.drop(train_df.index)

    # Make sure that the validation set does not have labels that were not included in train
    valid_df_labels = train_df.label.unique()
    valid_df = valid_df[valid_df.label.isin(valid_df_labels)]

    return train_df, valid_df

def bad_char_del(text):
     return text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
