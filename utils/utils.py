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
    except Exception as err:
        raise Exception("Random seed has not been set yet. Please set before running.")

def set_random_seed(seed_value):
    os.environ[seed_env_var_name] = str(seed_value)

def randomly_shuffle_list(list):
    random.seed(a=get_random_seed())
    random.shuffle(list)

def bad_char_del(text):
     return text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
