import torch
import numpy as np

from tqdm import tqdm

from mapping.model_training.transformer_models import get_nsp_model_and_optimizer
from mapping.model_training.transformer_training_utils import get_tokenizer, check_best, load_model

from utils.utils import split_df

def train_nsp(df, params, device):
    # Trains a model to predict whether the supplied sentence is the next sentence

    # Get the name of the model that we want to load
    model_name = params["model_name"]

    # Get the tokenizer for this model
    tokenizer = get_tokenizer(model_name)

    # Get a nsp language model and optimizer
    model, optimizer = get_nsp_model_and_optimizer(params, device)

    # Split our data into train and validate
    train_df, val_df = split_df(df, split_frac=0.7, has_labels=False)

    # Create dataloaders for our training sets
    train_dataloader = get_nsp_dataloader(train_df, tokenizer, params, is_train=True)
    val_dataloader = get_nsp_dataloader(val_df, tokenizer, params, is_train=False)

    model = train_nsp_model(model, train_dataloader, val_dataloader, optimizer, device, params)

    if "bert" == model_name[:4]:
        model = model.bert
    else:
        raise Exception(f"{model_name} not currently supported in training nsp language models")

    return model

def get_nsp_dataloader(df, tokenizer, params, is_train):
    # Creates a dataloader for all classification datasets
    max_len = params["max_length"]
    batch_size = params["batch_size"]

    # Tokenize and convert to input IDs
    input_ids, token_type_ids, attention_mask = get_nsp_inputs(df.first_text, df.second_text, tokenizer, max_len)

    y = torch.LongTensor(np.stack(df.label.values))

    data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(input_ids, token_type_ids, attention_mask, y), batch_size=batch_size, shuffle=is_train)

    return data_loader

# Turns the text series into a PyTorch tensor
def get_nsp_inputs(first_text_series, second_text_series, tokenizer, max_length):
    # Make a list to tuples, with each has a first_text and second_text
    combined_list = [(f, s) for f, s in zip(first_text_series, second_text_series)]

    tensors = tokenizer.batch_encode_plus(combined_list, max_length = max_length, pad_to_max_length=True, truncation=True, return_tensors="pt")

    input_ids = tensors["input_ids"]
    token_type_ids = tensors["token_type_ids"]
    attention_mask = tensors["attention_mask"]

    return input_ids, token_type_ids, attention_mask

def train_nsp_model(model, train_dl, val_dl, optim, device, params):
    # Get epoch number and patience for this training
    epochs = params["epochs"]
    patience = params["patience"]

    # Initialize epochs since last best and initial best score
    epochs_since_last_best = 0
    best_score = -1

    # Iterate over n epochs
    for epoch in tqdm(range(epochs), desc="Epoch"):

        # Train on all batches
        train_progress = tqdm(train_dl, desc="Batch")
        for input_ids, type_ids, att_mask, y in train_progress:

            loss = train_on_nsp_batch(model, input_ids, type_ids, att_mask, y, optim, device)
            train_progress.set_description(f"Masking batch loss - {loss}")

        inverse_loss = get_val_nsp_inverse_loss(model, val_dl, device)

        # Check if the average target metric has improved since the last best. If so, save model.
        # Also check if the patience for training has been exceeded.
        is_patience_up, epochs_since_last_best, best_score = check_best(model, epochs_since_last_best, inverse_loss, best_score, patience)

        if is_patience_up:
            print(f"Ceasing training after {epoch} epochs with the best score at {best_score}")
            break

    # Load the best saved model
    load_model(model)

    return model

def get_val_nsp_inverse_loss(model, val_dl, device):
    # Gets the inverse of the loss of the whole dataset over the validation set.
    # We take the inverse loss as we want the best_model functions to work with average f1 score etc. as well.
    # For Average F1 score, bigger is better, so we inverse the loss such that bigger is better for that too.
    total_loss = 0

    # Turn model onto evaluation mode
    model.eval()

    # Validate on all batches
    val_progress = tqdm(val_dl, desc="Validation batch")
    for input_ids, type_ids, att_mask, y in val_progress:

        with torch.no_grad():
            loss = get_nsp_model_loss(model, input_ids, type_ids, att_mask, y, device)
            loss = loss.item()
            total_loss += loss

        val_progress.set_description(f"Inverse validation total loss {1/total_loss}")

    # Turn model back onto training mode
    model.train()

    inverse_loss = 1/total_loss

    return inverse_loss

def get_nsp_model_loss(model, input_ids, type_ids, att_mask, y, device):
    # Get the Xs onto device
    input_ids = input_ids.to(device)
    type_ids = type_ids.to(device)
    att_mask = att_mask.to(device)
    y = y.to(device)

    # Put the Xs into the model as both input and output
    outputs = model(input_ids = input_ids,
                    token_type_ids = type_ids,
                    attention_mask = att_mask,
                    next_sentence_label=y)

    # Loss is calculated in the model using CrossEntropyLoss
    loss = outputs[0]

    return loss

def train_on_nsp_batch(model, input_ids, type_ids, att_mask, y, optim, device):
    # Loss is calculated in the model using CrossEntropyLoss
    loss = get_nsp_model_loss(model, input_ids, type_ids, att_mask, y, device)

    #Backpropagate the error through the model
    loss.backward()
    optim.step()
    # Reset the gradient of the model
    model.zero_grad()
    optim.zero_grad()

    return loss.item()
