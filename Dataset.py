import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def read_data(file, task):
    """
    Read the csv file and return the data.
    Arguments:
        file(str): File name path.
        task(int): Task identifier(1 or 2).
    Returns:
        List of reviews' id.
        List of reviews.
        DataFrame of labeled aspects.
    """

    assert task in [1, 2], "Task must be 1 or 2"
    df = pd.read_csv(file)

    """ Processing the missing value """
    df.replace("", np.nan, inplace=True)
    print(f"[INFO] File: {file.ljust(9, ' ')}, Total missing value: {pd.isnull(df).any(axis=1).sum()}")
    df.fillna(value=0, inplace=True)

    aspect_col = [cat for cat in df.columns if cat not in ["id", "review"]]
    # Task 1 -> Replace values(0: Not mentioned; 1: Mentioned).
    if task == 1:
        replace_values = {-2: 0, -1: 1, 0: 1, 1: 1}
    # Task 2 -> Replace values(0: Not mentioned; 1: Negative; 2: Neutral; 3: Positive).
    else:
        replace_values = {1: 3, 0: 2, -1: 1, -2: 0}

    return list(df["id"]), list(df["review"]), df[aspect_col].replace(replace_values)

""" Preprocess the reviews """
def processing(review_list, replace_symbols):
    """
    Replace all the reviews by given symbol table.
    Arguments:
        review_list(list): List of reviews.
        replace_symbols(dict): Replaced symbol table.
    Returns:
        List of cleaning reviews.
    """
    review_clean = []
    for review in review_list:
        for symbol in replace_symbols.items():
            review = review.replace(*symbol)
        review_clean.append(review)
    return review_clean


""" Get train / dev / test dataset. """
def get_dataset(
        train_reviews_clean,        # Training reviews.
        train_sentiments,           # Training labeled data.
        dev_reviews_clean,          # Deviation reviews.
        dev_sentiments,             # Deviation labeled data.
        test_reviews_clean,         # Testing reviews.
        test_sentiments,            # Testing labeled data(Empty).
        tokenizer,                  # Pretrained tokenizer.
        config                      # Config file.
    ):
    max_seq_len = config.max_seq_len        # Maximum sequence length.

    train_dataset = ReviewDataset("train", train_reviews_clean, train_sentiments.values.tolist(), tokenizer, max_seq_len=max_seq_len)
    dev_dataset   = ReviewDataset("dev", dev_reviews_clean, dev_sentiments.values.tolist(), tokenizer, max_seq_len=max_seq_len)
    test_dataset  = ReviewDataset("test", test_reviews_clean, test_sentiments.values.tolist(), tokenizer, max_seq_len=max_seq_len)
    return train_dataset, dev_dataset, test_dataset

class ReviewDataset(Dataset):
    def __init__(self, split, reviews, aspects, tokenizer, max_seq_len=512):
        self.split = split                  # train / dev / test.
        self.reviews = reviews              # List of reviews.
        self.aspects = aspects              # List of labeled aspects.
        self.tokenizer = tokenizer          # Tokenizer.
        self.max_seq_len = max_seq_len      # Maximum sequence length.

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]

        # Encoding the review.
        inputs = self.tokenizer.encode_plus(
                    review,
                    add_special_tokens=True,
                    max_length=self.max_seq_len,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                )

        # Get encoding and masking.
        ids = inputs["input_ids"].flatten()
        mask = inputs["attention_mask"].flatten()

        # Training / Validation
        if self.split in ["train", "dev"]:
            aspect = self.aspects[idx]
            return{
                "ids": torch.as_tensor(ids, dtype=torch.long),
                "mask": torch.as_tensor(mask, dtype=torch.long),
                "labels": torch.as_tensor(aspect, dtype=torch.long)
            }
        # Testing
        else:
            return{
                "ids": torch.as_tensor(ids, dtype=torch.long),
                "mask": torch.as_tensor(mask, dtype=torch.long),
            }