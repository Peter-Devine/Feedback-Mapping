import torch

from tqdm import tqdm

from mapping.model_training.transformer_models import get_masking_model_and_optimizer
from mapping.model_training.transformer_training_utils import get_inputs, get_tokenizer, check_best, load_model

from utils.utils import split_df

def train_mask(df, params, device):
    # Trains a model to predict masked words on inputted text

    # Get the name of the model that we want to load
    model_name = params["model_name"]

    # Get the tokenizer for this model
    tokenizer = get_tokenizer(model_name)

    # Get a masked language model and optimizer
    model, optimizer = get_masking_model_and_optimizer(params, device)

    # Split our data into train and validate
    train_df, val_df = split_df(df, split_frac=0.7, has_labels=False)

    # Create dataloaders for our training sets
    train_dataloader = get_masking_dataloader(train_df, tokenizer, params, is_train=True)
    val_dataloader = get_masking_dataloader(val_df, tokenizer, params, is_train=False)

    model = train_masking_model(model, train_dataloader, val_dataloader, optimizer, device, params)

    if "roberta" in model_name:
        model = model.roberta
    elif "albert" in model_name:
        model = model.albert
    elif "bert" in model_name:
        model = model.bert
    else:
        raise Exception(f"{model_name} not currently supported in training masking language models")

    return model

def get_masking_dataloader(df, tokenizer, params, is_train):
    # Creates a dataloader for all classification datasets
    max_len = params["max_length"]
    batch_size = params["batch_size"]

    # Tokenize and convert to input IDs
    X = get_inputs(df.text, tokenizer, max_len)

    data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X), batch_size=batch_size, shuffle=is_train)

    return data_loader

def train_masking_model(model, train_dl, val_dl, optim, device, params):
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
        for input_ids in train_progress:

            loss = train_on_mask_batch(model, input_ids, optim, device)
            train_progress.set_description(f"Masking batch loss - {loss}")

        inverse_loss = get_val_mask_inverse_loss(model, val_dl, device)

        # Check if the average target metric has improved since the last best. If so, save model.
        # Also check if the patience for training has been exceeded.
        is_patience_up, epochs_since_last_best, best_score = check_best(model, epochs_since_last_best, inverse_loss, best_score, patience)

        if is_patience_up:
            print(f"Ceasing training after {epoch} epochs")
            break

    # Load the best saved model
    load_model(model)

    return model

def get_val_mask_inverse_loss(model, val_dl, device):
    # Gets the inverse of the loss of the whole dataset over the validation set.
    # We take the inverse loss as we want the best_model functions to work with average f1 score etc. as well.
    # For Average F1 score, bigger is better, so we inverse the loss such that bigger is better for that too.
    total_loss = 0

    # Turn model onto evaluation mode
    model.eval()

    # Validate on all batches
    val_progress = tqdm(val_dl, desc="Validation batch")
    for input_ids in val_progress:

        with torch.no_grad():
            loss = get_mask_model_loss(model, input_ids, device)
            loss = loss.item()
            total_loss += loss

    # Turn model back onto training mode
    model.train()

    inverse_loss = 1/total_loss
    val_progress.set_description(f"Total loss {total_loss}")
    return inverse_loss

def get_mask_model_loss(model, input_ids, device):
    # Dataloader stores data in a list, but in this case, we only have one  input stored (I.e. we have X, but no y).
    # In short, input_ids is initially a list with one element, so we just pop that out of its element
    input_ids = input_ids[0]

    # Get the Xs onto device
    input_ids = input_ids.to(device)

    # Put the Xs into the model as both input and output
    outputs = model(input_ids, labels=input_ids)

    # Loss is calculated in the model using CrossEntropyLoss
    loss = outputs[0]

    return loss

def train_on_mask_batch(model, input_ids, optim, device):
    # Loss is calculated in the model using CrossEntropyLoss
    loss = get_mask_model_loss(model, input_ids, device)

    #Backpropagate the error through the model
    loss.backward()
    optim.step()
    # Reset the gradient of the model
    model.zero_grad()
    optim.zero_grad()

    return loss.item()
