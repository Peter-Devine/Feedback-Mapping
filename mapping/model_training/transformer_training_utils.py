import torch
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration

import numpy as np

import os

from tqdm import tqdm

from utils.utils import randomly_shuffle_list

def get_lm_and_tok(model_name, device):
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Load the BERT model
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model.zero_grad()

    return model, tokenizer

# Turns the text series into a PyTorch tensor
def get_inputs(text_series, tokenizer, max_length):
    return tokenizer.batch_encode_plus(list(text_series.values), max_length = max_length, pad_to_max_length=True, truncation=True, return_tensors="pt")["input_ids"]

# Takes a dictionary of tasks, and returns a list of task names proportional to the data size of each task, shufffled.
# E.g. ["TaskB", "TaskB", "TaskA", "TaskC", "TaskB", "TaskC"] where TaskA has 1 batch of data, TaskB has 3 batches of data, and Task C has 2 batches of data
def get_mtl_task_order(tasks_dict):
    # Init empty list
    task_training_list = []

    for task_name, task_data in tasks_dict.items():
        # Fill up list with repetitions of task name string equal to the number of batches for that task
        number_of_observations = len(task_data["train_dataloader"])
        task_training_list.extend([task_name]*number_of_observations)

    # Shuffle list
    randomly_shuffle_list(task_training_list)

    return task_training_list

# Takes a dictionary of tasks and returns a dictionary of iterable training dataloaders for each task, so that we do not need to iterate through an entire dataloader at once.
def get_mtl_train_iters(tasks_dict):
    # Start with empty dict
    training_iters = {}
    for task_name, task_data in tasks_dict.items():
        # Get an iterable for each task, which we will refresh each epoch. This allows us to jump between tasks, so we dont have to continually iterate over one dataloader
        training_iters[task_name] = iter(task_data["train_dataloader"])

    return training_iters

# Takes the task dict for a given MTL training, and returns the validations scores for each task
def get_mtl_validation_results(tasks_dict):
    task_results = {}
    for task_name, task_data in tasks_dict.items():
        # Get the training model
        model = task_data["training_model"]
        # Set the model to eval mode (I.e. stop dropout etc.)
        model = model.eval()

        # Get the eval engine specific to this model
        eval_engine = task_data["eval_engine"]

        # Get the validation set dataloader
        val_dataloader = task_data["val_dataloader"]

        # Run validation on the above model using the validation set
        results = eval_engine.run(val_dataloader).metrics

        # Revert model to training mode
        model = model.train()

        # Add results to output dict
        task_results[task_name] = results

    return task_results

def train_on_mtl_datasets(tasks_dict, loss_fn, params, device, target_metric):
    # Get epoch number and patience for this training
    epochs = params["epochs"]
    patience = params["patience"]

    # Initialize epochs since last best and initial best score
    epochs_since_last_best = 0
    best_score = -1

    for epoch in tqdm(range(epochs), desc="Epoch"):

        # Get a list of all tasks to train on, with one entry per batch that that task has in its training set. Then randomly shuffle these tasks, and train in that order.
        task_training_list = get_mtl_task_order(tasks_dict)
        # Get a dict of iterable dataloaders, one for each task
        training_iters = get_mtl_train_iters(tasks_dict)

        # Train on all batches
        tasks_progress = tqdm(task_training_list, desc="Batch")
        for task_name in tasks_progress:
            # Get the current batch and relevant data for the current training task
            batch = next(training_iters[task_name])
            model = tasks_dict[task_name]["training_model"]
            optim = tasks_dict[task_name]["optimizer"]
            n_classes = tasks_dict[task_name]["n_classes"]

            loss = train_on_batch(model, batch, optim, loss_fn, n_classes, device)
            tasks_progress.set_description(f"Batch loss at {task_name} - {loss}")

        task_results = get_mtl_validation_results(tasks_dict)

        # Calculate whether patience has been exceeded or not on the target metric
        # Here, we average the target metric across all datasets
        target_score_list = [results[target_metric] for task_name, results in task_results.items()]
        target_score = sum(target_score_list) / len(target_score_list)
        print(f"{target_metric} is {target_score} at epoch {epoch}")

        # Get any model from tasks dict. They all share the same shared language model layer anyway, which is the one we will take.
        any_model = get_any_model_from_mtl_task_dict(tasks_dict)
        # Check if the average target metric has improved since the last best. If so, save model.
        # Also check if the patience for training has been exceeded.
        is_patience_up, epochs_since_last_best, best_score = check_best(any_model, epochs_since_last_best, target_score, best_score, patience)

        # When patience has been exceeded, cease training, ad the model is no longer getting any better.
        if is_patience_up:
            print(f"Stopping training at epoch {epoch} with best metric as {best_score}")
            break

    any_model = get_any_model_from_mtl_task_dict(tasks_dict)
    # Loading model from temporary storage
    load_model(any_model)

    return any_model

def train_on_batch(model, batch, optim, loss_fn, n_classes, device):
    # Get the Xs and Ys
    inputs = batch[:-1]
    golds = batch[-1].to(device)

    # Put the Xs into the model
    logits = model(inputs)

    # Get the loss between the model outputs and the gold values
    loss = loss_fn(logits.view(-1, n_classes), golds.view(-1))

    #Backpropagate the error through the model
    loss.backward()
    optim.step()
    # Reset the gradient of the model
    model.zero_grad()
    optim.zero_grad()

    return loss.item()

def check_best(model, epochs_since_last_best, target_score, best_score, patience):
    # First, check if the new validation score is better than the previous best
    if target_score > best_score:
        # If the score is better than previous best, save the current model as the new best model.
        # Reset the "since last best" counter
        best_score = target_score
        epochs_since_last_best = 0
        print(f"Saving new best model (Score: {target_score})")
        save_model(model)
    else:
        # If not, iterate the since last best counter
        epochs_since_last_best += 1

    # If the number of epochs since last best exceeds the training patience, then we set is_patience_up = True, which stops training
    if epochs_since_last_best > patience:
        is_patience_up = True
    else:
        is_patience_up = False

    return is_patience_up, epochs_since_last_best, best_score

# Gets an arbitrary model from the tasks_dict, as we only want the language model from any given model, which is the same for every tasks model. All task's models share the same language model.
def get_any_model_from_mtl_task_dict(tasks_dict):
    first_task = list(tasks_dict.keys())[0]
    return tasks_dict[first_task]["training_model"].lang_model

def setup_temp_model_repo():
    temp_model_repo_path = os.path.join(".", "mapping", "model_training", "temp_models")
    if not os.path.exists(temp_model_repo_path):
        print(f"Creating {temp_model_repo_path}...")
        os.mkdir(temp_model_repo_path)
    return temp_model_repo_path

def delete_temp_model(model_name = "temp"):
    temp_model_repo_path = setup_temp_model_repo()
    temp_model_path = os.path.join(temp_model_repo_path, f"{model_name}.pt")
    print(f"Deleting {temp_model_path}...")
    os.remove(temp_model_path)

def save_model(model, model_name = "temp"):
    temp_model_repo = setup_temp_model_repo()
    temp_model_path = os.path.join(temp_model_repo, f"{model_name}.pt")

    print(f"Saving temp model at {temp_model_path}")
    torch.save(model.state_dict(), temp_model_path)

def load_model(model, model_name = "temp"):
    temp_model_repo = setup_temp_model_repo()
    temp_model_path = os.path.join(temp_model_repo, f"{model_name}.pt")

    print(f"Loading temp model from {temp_model_path}")
    model.load_state_dict(torch.load(temp_model_path))

    delete_temp_model()
