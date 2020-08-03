import torch
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration

import numpy as np

import os

from tqdm import tqdm

from mapping.model_training.transformer_models import get_cls_model_and_optimizer, get_nsp_model_and_optimizer, get_weighted_adam_optimizer
from mapping.model_training.transformer_eval import create_eval_engine

from utils.utils import randomly_shuffle_list, split_df

def train_cls(data_dict, params, device, training_type="cls"):
    # Three training types are allowed:
    # Classification - taking one piece of text and classifying it given a set of labels
    # Similarity classification - given two pieces of text, classifying whether they are similar or not, in a binary fashion (E.g. next sentence prediction, duplicate issues etc.)
    allowed_training_types = ["cls", "sim_cls"]
    assert training_type in allowed_training_types, f"Cannot train using {training_type}. Currently only {allowed_training_types} are supported."

    # Get the hyperparameters for training
    lr = params["lr"]
    eps = params["eps"]
    wd = params["wd"]
    epochs = params["epochs"]
    patience = params["patience"]
    max_len = params["max_len"]
    batch_size = params["batch_size"]
    model_name = params["model_name"]

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Load the BERT model
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model.zero_grad()

    tasks_dict = {}

    for task_name, train_df in data_dict.items():
        train_df, val_df = split_df(train_df)

        # Get the dataloader for both training and test sets
        # This dataloader holds the inputs and outputs of each observation, and batches them
        if training_type == "cls":
            train_loader, label_dict = get_cls_dataloader(train_df, tokenizer, max_len, batch_size, is_train=True, label_dict=None)
            val_loader, _ = get_cls_dataloader(val_df, tokenizer, max_len, batch_size, is_train=False, label_dict=label_dict)
            n_classes = len(label_dict.keys())
        elif training_type == "sim_cls":
            train_loader = get_sim_cls_dataloader(train_df, tokenizer, max_len, batch_size, is_train=True)
            val_loader = get_sim_cls_dataloader(val_df, tokenizer, max_len, batch_size, is_train=False)
            n_classes = 2
        else:
            raise Exception(f"Cannot train using {training_type}. Currently only {allowed_training_types} are supported.")

        # Prepare the model and optimizer
        if training_type == "cls":
            training_model, optimizer = get_cls_model_and_optimizer(model, n_classes, lr, eps, wd, device)
        elif training_type == "sim_cls":
            training_model, optimizer = get_nsp_model_and_optimizer(model, lr, eps, wd, device)

        eval_engine = create_eval_engine(training_model, device)

        tasks_dict[task_name] = {"train_dataloader": train_loader,
                            "val_dataloader": val_loader,
                            "n_classes": n_classes,
                            "training_model": training_model,
                            "optimizer": optimizer,
                            "eval_engine": eval_engine}

    # Prepare the loss function
    if training_type in ["cls", "sim_cls"]:
        loss_fn = torch.nn.CrossEntropyLoss()
        target_metric = "average f1"
    else:
        # If more training types are added, add appropriate loss fn
        pass

    model = train_on_datasets(tasks_dict, loss_fn, epochs, patience, device, target_metric)

    return model

def get_cls_dataloader(df, tokenizer, max_len, batch_size, is_train, label_dict=None):
    # Creates a dataloader for all classification datasets

    # Tokenize and convert to input IDs
    X = get_inputs(df.text, tokenizer, max_len)
    # Get labels for each observation
    y, label_dict = get_labels(df.label, label_dict=label_dict)

    data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=batch_size, shuffle=is_train)

    return data_loader, label_dict

def get_sim_cls_dataloader(df, tokenizer, max_len, batch_size, is_train):
    # Creates a dataloader for all similarity classification (similar/not similar binary classification) datasets

    # Tokenize and convert to input IDs
    X_first = get_inputs(df.first_text, tokenizer, max_len)
    X_second = get_inputs(df.second_text, tokenizer, max_len)

    # Get labels for each observation
    y = torch.LongTensor(np.stack(df.label.values))

    data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_first, X_second, y), batch_size=batch_size, shuffle=is_train)

    return data_loader

def get_input_label_text_dataloader(df, tokenizer, max_len, batch_size, is_train):
    # Creates a dataloader for input/output text dataset

    # Tokenize and convert to input IDs
    input_ids = get_inputs(df.input_text, tokenizer, max_len)
    label_ids = get_inputs(df.output_text, tokenizer, max_len)

    data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(input_ids, label_ids), batch_size=batch_size, shuffle=is_train)

    return data_loader

def get_inputs(text_series, tokenizer, max_len):
    return tokenizer.batch_encode_plus(list(text_series.values), max_length = max_len, pad_to_max_length=True, truncation=True, return_tensors="pt")["input_ids"]

def get_labels(labels, label_dict=None):
    if label_dict is None:
        label_dict = {label: i for i, label in enumerate(labels.unique())}
    int_labels = labels.apply(lambda x: label_dict[x])
    return torch.LongTensor(np.stack(int_labels.values)), label_dict

def train_on_datasets(tasks_dict, loss_fn, epochs, patience, device, target_metric):

    epochs_since_last_best = 0
    best_score = -1

    for epoch in tqdm(range(epochs), desc="Epoch"):

        # Get a list of all tasks to train on, with one entry per batch that that task has in its training set. Then randomly shuffle these tasks, and train in that order.
        task_training_list = []
        training_iters = {}
        for task_name, task_data in tasks_dict.items():
            number_of_observations = len(task_data["train_dataloader"])
            task_training_list.extend([task_name]*number_of_observations)

            # Get an iterable for each task, which we will refresh each epoch. This allows us to jump between tasks, so we dont have to continually iterate over one dataloader
            training_iters[task_name] = iter(task_data["train_dataloader"])
        randomly_shuffle_list(task_training_list)

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

        task_results = {}
        for task_name, task_data in tasks_dict.items():
            # Eval on validation set
            model = task_data["training_model"]
            model = model.eval()

            eval_engine = task_data["eval_engine"]
            val_dataloader = task_data["val_dataloader"]

            results = eval_engine.run(val_dataloader).metrics
            task_results[task_name] = results
            model = model.train()

        # Calculate whether patience has been exceeded or not on the target metric
        target_score = sum([results[target_metric] for task_name, results in task_results.items()]) / len(task_results.keys())
        print(f"{target_metric} is {target_score} at epoch {epoch}")

        # Get any model from tasks dict. They all share the same shared language model layer anyway, which is the one we will take.
        any_model = get_any_model_from_task_dict(tasks_dict)
        is_patience_up, epochs_since_last_best, best_score = check_best(any_model, epochs_since_last_best, target_score, best_score, patience)

        # Calculate whether patience has been exceeded or not
        if is_patience_up:
            print(f"Stopping training at epoch {epoch} with best metric as {best_score}")
            break

    any_model = get_any_model_from_task_dict(tasks_dict)
    # Loading model from temporary storage
    load_model(any_model)

    return any_model.lang_model

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

    if target_score > best_score:
        best_score = target_score
        epochs_since_last_best = 0
        print(f"Saving new best model (Score: {target_score})")
        save_model(model)
    else:
        epochs_since_last_best += 1

    if epochs_since_last_best > patience:
        is_patience_up = True
    else:
        is_patience_up = False

    return is_patience_up, epochs_since_last_best, best_score

def get_any_model_from_task_dict(tasks_dict):
    first_task = list(tasks_dict.keys())[0]
    return tasks_dict[first_task]["training_model"]

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
