import re
import math
import pandas as pd
import random
from transformers import AutoTokenizer
from nltk.util import ngrams
from utils.utils import get_random_seed, split_df, randomly_sample_list

################ CURRENTLY SIMILARITY SCORES ARE UNUSED, BUT MIGHT BE HANDY LATER ###############################
#
# # Get the overlap coefficient (https://en.wikipedia.org/wiki/Overlap_coefficient) of two given sets
# def get_overlap_coefficient(set1, set2):
#     return float(len(set1 & set2)) / min(len(set1), len(set2))
#
# # Get the Jaccard index (https://en.wikipedia.org/wiki/Jaccard_index) of two given sets
# def get_jaccard_index(set1, set2):
#     return float(len(set1 & set2)) / float(len(set1 | set2))
#
# # Get the Ochiai coefficient (cos similarity) (https://en.wikipedia.org/wiki/Cosine_similarity#Otsuka-Ochiai_coefficient) of two given sets
# def get_ochiai_coefficient(set1, set2):
#     return float(len(set1 & set2)) / (math.sqrt(float(len(set1))) *
#                                           math.sqrt(float(len(set2))))
#
# # Get the Dice score (https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) of two given sets
# def get_dice_score(set1, set2):
#     return 2.0 * float(len(set1 & set2)) / float(len(set1) + len(set2))

def get_shuffled_second_text(split_df):
    # Take in a df of first and second texts, shuffled it, shifts the second text only by one and then outputs both texts
    # This ensures that we have randomly shuffled pairs of texts that are always paired with a different other text

    # Do the shuffle
    split_df = split_df.sample(frac=1, random_state = get_random_seed())

    # Shift all second texts along one
    new_second_texts = split_df.second_text.shift(1)
    new_second_texts.iloc[0] = split_df.second_text.iloc[-1]

    # Make this shifted second_text the new second_text (I.e. it is not matched with first_text and more)
    split_df["second_text"] = new_second_texts
    return split_df

def get_split(row, is_one_sentence=False):
    # Find splits by newline OR word,punctuation,space for normal datasets or word,space for pre-sentenced datasets
    if is_one_sentence:
        search_regex = "[\n]|([\w][\s])"
    else:
        search_regex = "[\n]|([\w][!?.][\s])"

    # Find all possible splits in the given text
    all_possible_splits_idx = re.finditer(search_regex, row)

    possible_splits = []

    for split in all_possible_splits_idx:
        split_idx = split.end()
        presplit = row[:split_idx-1] # -1 to remove the splitting character from the split itself
        postsplit = row[split_idx:]

        # Check that each split (both before and after split strings) contain at least MIN_WORDS number of words.
        MIN_WORDS = 3
        if len(presplit.split()) >= MIN_WORDS and len(postsplit.split()) >= MIN_WORDS:
            possible_splits.append((presplit, postsplit))

    num_good_splits = len(possible_splits)

    # If we do not have any good splits (split by punctuation), then we try to split by whitespace.
    if num_good_splits < 1 and is_one_sentence:
        return []
    elif num_good_splits < 1:
        return get_split(row, is_one_sentence=True)

    # Get one split, randomly selected from the possible random splits
    random_split = randomly_sample_list(possible_splits, k=1)[0]

    return random_split


def get_two_text_df(split_df):
    first_splits = []
    second_splits = []
    ids = []

    # We only want to iterate over observations with splits available
    split_exists_mask = split_df.str.len() > 0

    for split in split_df[split_exists_mask]:
        first_splits.append(split[0])
        second_splits.append(split[1])

    two_text_pd = pd.DataFrame({"first_text": first_splits,
                 "second_text": second_splits})

    return two_text_pd

def get_next_sentence_df(df):
    # First, remove any duplicate text entries in the df
    df = df[~df.text.duplicated()]

    # Split each line at a suitable place. Randomly sample one valid split for each line from all possible splits.
    all_split_df = df.text.apply(get_split)

    # Make these splits into a dataframe with columns "first_text" and "second_text"
    matched_df = get_two_text_df(all_split_df)

    # Randomly split the dataset into two
    first_split_df, second_split_df = split_df(matched_df, split_frac=0.5, has_labels=False)

    # Get both sides of the split, and randomly shuffle one of them so that they are no longer paired with paired text
    unmatched_df = get_shuffled_second_text(second_split_df)
    # Keep one half of the split paired
    matched_df = first_split_df

    # Add labels to dataframe to train model
    matched_df["label"] = 1
    unmatched_df["label"] = 0

    # Append matched and unmatched data
    full_df = matched_df.append(unmatched_df).reset_index(drop=True)

    return full_df
