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

def get_next_sentence_df(df):

    df = df[~df.text.duplicated()]

    all_split_df = df.text.apply(get_splits)

    matched_df = get_two_text_df(all_split_df)

    # Maybe comment this line out, might make training better, as we will have more examples of splits
    matched_df = matched_df.groupby('id', group_keys=False).apply(lambda df: df.sample(1, random_state = get_random_seed()))

    # Again, maybe comment below out to allow the same sentence to be matched and unmatched
    unmatched_df = matched_df.sample(frac=0.5, random_state = get_random_seed())
    matched_df = matched_df.drop(unmatched_df.index)

    unmatched_second_text_df = unmatched_df.apply(lambda x: unmatched_df[unmatched_df.label != x["id"]].sample(1, random_state = get_random_seed()).iloc[0], axis=1)

    unmatched_df["second_text"] = unmatched_second_text_df["second_text"]

    unmatched_df["label"] = 0
    matched_df["label"] = 1

    full_df = matched_df.append(unmatched_df).reset_index(drop=True)

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
                 "id": ids}, index = ids)

    return two_text_pd
