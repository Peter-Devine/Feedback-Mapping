import re
import pandas as pd
from utils.utils import get_random_seed

def get_splits(row):

    all_possible_splits_idx = re.finditer("[\n]|([\w][!?.][\s])", row)

    possible_splits = []

    for i, split in enumerate(all_possible_splits_idx):
        split_idx = split.end()
        presplit = row[:split_idx-1]
        postsplit = row[split_idx:]

        MIN_WORDS = 3

        if len(presplit.split()) >= MIN_WORDS and len(postsplit.split()) >= MIN_WORDS:
            possible_splits.append((presplit, postsplit))

    return possible_splits

def pair_class_df(class_df, class_column="label"):
    # Pairs senetences together that are from the same class

    paired_df = None

    # Iterate through all the unique classes, creating an array of paired texts for each class
    for unique_class in class_df[class_column].unique():
        # Get all the texts from a given class, and shuffle them
        first_texts = class_df[class_df[class_column]==unique_class].text.sample(frac=1)
        # Shift all these shuffled texts along one
        second_texts = first_texts.shift(1)
        second_texts.iloc[0] = first_texts.iloc[-1]

        # Paired both the shifted and unshifted texts together (Thus we make sure that each text has a different corresponding pair)
        single_class_paired_df = pd.DataFrame({"first_text": first_texts, "second_text": second_texts, "id": unique_class})

        # Add this paired df for one class to overall paired df
        if paired_df is None:
            paired_df = single_class_paired_df
        else:
            paired_df = paired_df.append(single_class_paired_df)

    paired_df = paired_df.reset_index(drop=True)

    return paired_df


def shuffle_paired_df(matched_df):
    # Takes paired dataset, and makes half of it paired and the other half unpaired for training

    # First, split given dataset in two randomly
    unmatched_df = matched_df.groupby("id").apply(lambda x: x.sample(frac=0.5, random_state = get_random_seed())).droplevel(level=0)
    matched_df = matched_df.drop(unmatched_df.index)

    # Go through each unmatched df row, randomly sampling another row from the dataset that does not share an id with the current row
    unmatched_second_text_df = unmatched_df.apply(lambda x: unmatched_df[unmatched_df.id != x["id"]].sample(1, random_state = get_random_seed()).iloc[0], axis=1)

    # Set these sampled rows as the second text for the original row. These are hence unpaired as they do not share an id with the original row
    unmatched_df["second_text"] = unmatched_second_text_df["second_text"]

    # Set the matched/unmatched labels
    unmatched_df["label"] = 0
    matched_df["label"] = 1

    full_df = matched_df.append(unmatched_df).reset_index(drop=True)

    return full_df

def get_next_sentence_df(df):

    df = df[~df.text.duplicated()]

    all_split_df = df.text.apply(get_splits)

    matched_df = get_two_text_df(all_split_df)

    # Maybe comment this line out, might make training better, as we will have more examples of splits
    # matched_df = matched_df.groupby('id', group_keys=False).apply(lambda df: df.sample(1, random_state = get_random_seed()))

    # Again, maybe comment below out to allow the same sentence to be matched and unmatched
    full_df = shuffle_paired_df(matched_df)

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
