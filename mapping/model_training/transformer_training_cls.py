import torch
import numpy as np

from mapping.model_training.transformer_models import get_cls_model_and_optimizer
from mapping.model_training.transformer_eval import create_eval_engine
from mapping.model_training.transformer_training_utils import get_inputs, train_on_mtl_datasets, get_lm_and_tok

from utils.utils import split_df

def train_cls(data_dict, params, device):
    # Trains a model/models to take in text and predict a class from n_classes

    # Get the model and tokenizer that we will use across all training tasks
    model, tokenizer = get_lm_and_tok(model_name=params["model_name"], device=device)

    # Create a dict of tasks, which contains the model, data and other required data for training on that task
    tasks_dict = {}
    for task_name, train_df in data_dict.items():
        # First, split the data into a training and a validation set
        train_df, val_df = split_df(train_df)

        # Get the dataloader for both training and validation sets
        # This dataloader holds the inputs and outputs of each observation, and batches them
        train_loader, label_dict = get_cls_dataloader(train_df, tokenizer, params, is_train=True, label_dict=None)
        val_loader, _ = get_cls_dataloader(val_df, tokenizer, params, is_train=False, label_dict=label_dict)
        n_classes = len(label_dict.keys())

        # Prepare the model and optimizer
        training_model, optimizer = get_cls_model_and_optimizer(model, n_classes, params, device)

        # Get the eval engine for this task
        eval_engine = create_eval_engine(training_model, device)

        # Add the dataloaders, model, n_class information,
        tasks_dict[task_name] = {"train_dataloader": train_loader,
                            "val_dataloader": val_loader,
                            "n_classes": n_classes,
                            "training_model": training_model,
                            "optimizer": optimizer,
                            "eval_engine": eval_engine}

    # Prepare the loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # The metric that we will evaluate validation set performance on
    target_metric = "average f1"

    # Train language model on the dict of tasks, and output the final trained language model
    model = train_on_mtl_datasets(tasks_dict, loss_fn, params, device, target_metric)

    return model

def get_cls_dataloader(df, tokenizer, params, is_train, label_dict=None):
    # Creates a dataloader for all classification datasets
    max_len = params["max_len"]
    batch_size = params["batch_size"]

    # Tokenize and convert to input IDs
    X = get_inputs(df.text, tokenizer, max_len)
    # Get labels for each observation
    y, label_dict = get_labels(df.label, label_dict=label_dict)

    data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=batch_size, shuffle=is_train)

    return data_loader, label_dict
