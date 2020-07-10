import re
import math
import pandas as pd
from transformers import AutoTokenizer
from nltk.util import ngrams
from utils.utils import get_random_seed

def get_n_gram_sim_df(texts, n_gram_level, model_name, sim_type):
    # First, pair each text observation with a random other text observation
    first_texts, second_texts = get_shuffled_second_text(texts)

    # prepare the tokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Next, get the n-grams of each observation
    def get_series_n_grams(text_series):
        tokens = [tok.tokenize(x) for x in text_series]
        n_grams = [set(ngrams(x, n_gram_level)) for x in tokens]
        return n_grams

    first_n_grams = get_series_n_grams(first_texts)
    second_n_grams = get_series_n_grams(second_texts)

    # Finally, select a similarity scorer and apply it to each pair of n-grams
    SIM_SCORE_DICT = {
        "overlap": get_overlap_coefficient,
        "jaccard": get_jaccard_index,
        "ochiai": get_ochiai_coefficient,
        "dice": get_dice_score,
    }
    assert sim_type in SIM_SCORE_DICT.keys(), f"Invalid similarity score type {sim_type}. Please choose one of {SIM_SCORE_DICT.keys()}."
    sim_score = SIM_SCORE_DICT[sim_type]

    text_sim_scores = [sim_score(ngm1, ngm2) if len(ngm1) > 0 and len(ngm2) > 0 else 0 for ngm1, ngm2 in zip(first_n_grams, second_n_grams)]

    sim_df = pd.DataFrame({"first_text": first_texts, "second_text": second_texts, "score": text_sim_scores})

    return sim_df

# Get the overlap coefficient (https://en.wikipedia.org/wiki/Overlap_coefficient) of two given sets
def get_overlap_coefficient(set1, set2):
    return float(len(set1 & set2)) / min(len(set1), len(set2))

# Get the Jaccard index (https://en.wikipedia.org/wiki/Jaccard_index) of two given sets
def get_jaccard_index(set1, set2):
    return float(len(set1 & set2)) / float(len(set1 | set2))

# Get the Ochiai coefficient (cos similarity) (https://en.wikipedia.org/wiki/Cosine_similarity#Otsuka-Ochiai_coefficient) of two given sets
def get_ochiai_coefficient(set1, set2):
    return float(len(set1 & set2)) / (math.sqrt(float(len(set1))) *
                                          math.sqrt(float(len(set2))))

# Get the Dice score (https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) of two given sets
def get_dice_score(set1, set2):
    return 2.0 * float(len(set1 & set2)) / float(len(set1) + len(set2))

def get_shuffled_second_text(first_texts):
    # Take in a series of text, shuffled it, shifts it by one and then outputs both the shuffled and shifted texts
    # This ensures that we have randomly shuffled pairs of texts that are always paired with a different other text
    first_texts = first_texts.sample(frac=1, random_state = get_random_seed())
    second_texts = first_texts.shift(1)
    second_texts.iloc[0] = first_texts.iloc[-1]
    return first_texts, second_texts

def pair_class_df(class_df, class_column):
    # Pairs senetences together that are from the same class

    paired_df = None

    # Iterate through all the unique classes, creating an array of paired texts for each class
    for unique_class in class_df[class_column].unique():
        # Get all the texts from a given class, and shuffle them
        first_texts = class_df[class_df[class_column]==unique_class].text

        # Get a list of texts paired with a different other text (but they both have the same class)
        first_texts, second_texts = get_shuffled_second_text(first_texts)

        # Paired both the shifted and unshifted texts together (Thus we make sure that each text has a different corresponding pair)
        single_class_paired_df = pd.DataFrame({"first_text": first_texts, "second_text": second_texts, "id": unique_class})

        # Add this paired df for one class to overall paired df
        if paired_df is None:
            paired_df = single_class_paired_df
        else:
            paired_df = paired_df.append(single_class_paired_df)

    paired_df = paired_df.reset_index(drop=True)

    return paired_df

# Split the df randomly, and return both splits
def randomly_split_df(df, frac=0.5):
    first_df = df.sample(frac=0.5, random_state = get_random_seed())
    second_df = df.drop(first_df.index)
    return first_df, second_df

# Split df randomly, but make sure to keep the distribution of IDs (E.g. classes) the same for both splits
def randomly_split_df_id_stratified(df, frac=0.5):
    first_df = df.groupby("id").apply(lambda x: x.sample(frac=0.5, random_state = get_random_seed())).droplevel(level=0)
    second_df = df.drop(first_df.index)
    return first_df, second_df

def get_cls_pair_matched_df(class_df, class_column="label"):
    # First, pair each observation with another randomly chosen observation of the same class
    paired_df = pair_class_df(class_df, class_column)

    # Split these pairs, stratifying by class
    first_split_df, second_split_df = randomly_split_df_id_stratified(paired_df)

    # Then shuffle half of these pairs so that they are paired with another bit of text that is NOT a member of their class
    matched_and_unmatched_df = shuffle_paired_df(first_split_df, second_split_df)

    return matched_and_unmatched_df

def shuffle_paired_df(matched_df, unmatched_df):
    # Takes paired dataset, and makes half of it paired and the other half unpaired for training

    # Go through each unmatched df row, randomly sampling another row from the dataset that does not share an id with the current row
    unmatched_second_text_df = unmatched_df.apply(lambda x: unmatched_df[unmatched_df.id != x["id"]].sample(1, random_state = get_random_seed()).iloc[0], axis=1)

    # Set these sampled rows as the second text for the original row. These are hence unpaired as they do not share an id with the original row
    unmatched_df["second_text"] = unmatched_second_text_df["second_text"]

    # Set the matched/unmatched labels
    unmatched_df["label"] = 0
    matched_df["label"] = 1

    full_df = matched_df.append(unmatched_df).reset_index(drop=True)

    return full_df

def get_splits(row, is_one_sentence=False):
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
        presplit = row[:split_idx-1]
        postsplit = row[split_idx:]

        MIN_WORDS = 3

        if len(presplit.split()) >= MIN_WORDS and len(postsplit.split()) >= MIN_WORDS:
            possible_splits.append((presplit, postsplit))

    num_good_splits = len(possible_splits)

    if num_good_splits == 0 and not is_one_sentence:
        return get_splits(row, is_one_sentence=True)

    if is_one_sentence:
        # If there are no good splits even at a sentence level, then return an empty list
        if num_good_splits < 1:
            return []

        # Only return one mid-sentence split for each piece of text if it is just one sentence
        mid_sentence_split = possible_splits[int(num_good_splits/2)]
        return [mid_sentence_split]

    return possible_splits

def get_next_sentence_df(df):

    df = df[~df.text.duplicated()]

    all_split_df = df.text.apply(get_splits)

    matched_df = get_two_text_df(all_split_df)

    # Maybe comment this line out, might make training better, as we will have more examples of splits
    # Comment in if you want to have multiple instances of the same observation, just at different splits
    # matched_df = matched_df.groupby('id', group_keys=False).apply(lambda df: df.sample(1, random_state = get_random_seed()))

    # Randomly split the dataset into two
    first_split_df, second_split_df = randomly_split_df(matched_df)

    # Get both sides of the split, and randomly shuffle one of them so that they are no longer paired with paired text
    full_df = shuffle_paired_df(first_split_df, second_split_df)

    return full_df

def get_two_text_df(split_df):
    first_splits = []
    second_splits = []
    ids = []

    for i, splits in enumerate(split_df[split_df.str.len() > 0]):
        for split in splits:
            first_splits.append(split[0])
            second_splits.append(split[1])
            ids.append(i)

    two_text_pd = pd.DataFrame({"first_text": first_splits,
                 "second_text": second_splits,
                 "id": ids})

    return two_text_pd
