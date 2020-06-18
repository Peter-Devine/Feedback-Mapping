import torch
from transformers import AutoTokenizer, AutoModel

import os

from tqdm import tqdm

from mapping.model_training.transformer_models import get_cls_model_and_optimizer
from mapping.model_training.transformer_eval import create_eval_engine

def train_cls(train_df, val_df, model_name, batch_size, max_len, device):

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Load the BERT model
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model.zero_grad()

    def get_inputs(df):
        return tokenizer.batch_encode_plus(list(df.text.values), max_length = max_len, pad_to_max_length=True, return_tensors="pt")["input_ids"]

    def get_labels(df, label_dict = None):
        if label_dict is None:
            label_dict = {label: i for i, label in enumerate(df.label.unique())}
        int_labels = df.label.apply(lambda x: label_dict[x])
        return torch.LongTensor(np.stack(int_labels.values)), label_dict

    # Tokenize and convert to input IDs
    X_train = get_inputs(train_df)
    X_val = get_inputs(val_df)

    # Get labels for each observation
    y_train, label_dict = get_labels(train_df)
    y_val, _ = get_labels(val_df, label_dict=label_dict)

    # Batch tensor so we can iterate over inputs
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    n_classes = len(train_df.label.unique())
    lr = 5e-5
    eps = 1e-6
    wd = 0.01
    training_model, optimizer = get_cls_model_and_optimizer(model, n_classes, lr, eps, wd, device)

    loss_fn = torch.nn.CrossEntropyLoss()

    epochs = 3
    patience = 2
    model = train_on_dataset(training_model, train_loader, val_loader, optimizer, loss_fn, n_classes, epochs, patience, device)

    return model

def train_on_dataset(model, train, val, optim, loss_fn, n_classes, epochs, patience, device):

    TARGET_METRIC = "average f1"

    eval_engine = create_eval_engine(model, n_classes, device)

    epochs_since_last_best = 0
    best_score = 0

    for epoch in tqdm(range(epochs), desc="Epoch"):
        # Train on all batches
        model.train()
        batch_progress = tqdm(train, desc="Batch")
        for batch in batch_progress:
            loss = train_on_batch(model, batch, optim, loss_fn, n_classes, device)
            batch_progress.set_description(f"Batch loss {loss}")

        # Eval on validation set
        model.eval()
        results = eval_engine.run(val).metrics

        # Calculate whether patience has been exceeded or not on the target metric
        target_score = results[TARGET_METRIC]
        print(f"{TARGET_METRIC} is {target_score} at epoch {epoch}")

        is_patience_up, epochs_since_last_best, best_score = check_best(model, epochs_since_last_best, target_score, best_score, patience)

        # Calculate whether patience has been exceeded or not
        if is_patience_up:
            print(f"Stopping training at epoch {epoch} with best metric as {best_score}")
            break

    # Loading model from temporary storage
    load_model(model)

    return model.lang_model

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

def setup_temp_model_repo():
    temp_model_repo_path = os.path.join(".", "mapping", "model_training", "temp_models")
    if not os.path.exists(temp_model_repo_path):
        print(f"Creating {temp_model_repo_path}...")
        os.mkdir(temp_model_repo_path)
    return temp_model_repo_path

def delete_temp_models():
    temp_model_repo_path = setup_temp_model_repo()
    temp_model_path = os.path.join(temp_model_repo, "temp.pt")
    print(f"Deleting {temp_model_path}...")
    os.remove(temp_model_path)

def save_model(model):
    temp_model_repo = setup_temp_model_repo()
    temp_model_path = os.path.join(temp_model_repo, "temp.pt")

    print(f"Saving temp model at {temp_model_path}")
    torch.save(model.state_dict(), temp_model_path)

def load_model(model):
    temp_model_repo = setup_temp_model_repo()
    temp_model_path = os.path.join(temp_model_repo, "temp.pt")

    print(f"Loading temp model from {temp_model_path}")
    model.load_state_dict(torch.load(temp_model_path))

    delete_temp_models()

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

    return loss.item()
