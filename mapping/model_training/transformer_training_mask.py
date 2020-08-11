import torch

from tqdm import tqdm

from transformers import DataCollatorForLanguageModeling

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
    input_ids = get_inputs(df.text, tokenizer, max_len)
    masked_ids, labels = mask_ids(input_ids, tokenizer)

    data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(masked_ids, labels), batch_size=batch_size, shuffle=is_train)

    return data_loader

def mask_ids(input_ids, tokenizer):
    data_collator = DataCollatorForLanguageModeling(tokenizer)

    # DataCollatorForLanguageModeling needs a list of input ids - with one entry for every observation.
    list_of_ids = [ids for ids in input_ids]

    outputs = data_collator(list_of_ids)

    masked_input_ids = outputs["input_ids"]
    labels = outputs["labels"]

    return masked_input_ids, labels

def train_masking_model(model, train_dl, val_dl, optim, device, params):
    # Get epoch number and patience for this training
    epochs = params["epochs"]
    patience = params["patience"]

    # Initialize epochs since last best and initial best score
    epochs_since_last_best = 0
    best_score = -1

    # We log the training task (Always masking in this case) and the training size to identify exactly what task was getting done
    logger = TrainingLogger()
    logger.log(f"Training task: Masking")
    logger.log(f"Training size: {len(train_dl)}")
    logger.log(f"Val size: {len(val_dl)}\n")

    # Iterate over n epochs
    for epoch in tqdm(range(epochs), desc="Epoch"):

        # Train on all batches
        train_progress = tqdm(train_dl, desc="Batch")
        total_loss = 0
        for input_ids in train_progress:

            loss = train_on_mask_batch(model, input_ids, optim, device)
            train_progress.set_description(f"Masking batch loss - {loss}")
            total_loss += loss

        logger.log(f"Total training loss at epoch {epoch} is {total_loss}")

        inverse_loss = get_val_mask_inverse_loss(model, val_dl, device)

        logger.log(f"Inverse validation loss is {inverse_loss} at epoch {epoch}")

        # Check if the average target metric has improved since the last best. If so, save model.
        # Also check if the patience for training has been exceeded.
        is_patience_up, epochs_since_last_best, best_score = check_best(model, epochs_since_last_best, inverse_loss, best_score, patience)

        if is_patience_up:
            logger.log(f"Ceasing training after {epoch} epochs with the best score at {best_score}")
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

        val_progress.set_description(f"Inverse validation total loss {1/total_loss}")

    # Turn model back onto training mode
    model.train()

    inverse_loss = 1/total_loss
    return inverse_loss

def get_mask_model_loss(model, input_ids, device):
    # We have saved the masked ids as the first element, and the mask labels as the second
    masked_ids, labels = input_ids

    # Get the Xs onto device
    masked_ids = masked_ids.to(device)
    labels = labels.to(device)

    # Put the Xs into the model as both input and output
    outputs = model(masked_ids, labels=labels)

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
