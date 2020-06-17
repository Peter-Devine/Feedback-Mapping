import torch
from transformers import AutoTokenizer, AutoModel

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

    def get_labels(df):
        return torch.LongTensor(np.stack(df.label.values))

    # Tokenize and convert to input IDs
    X_train = get_inputs(train_df)
    X_val = get_inputs(val_df)

    # Get labels for each observation
    y_train = get_labels(train_df)
    y_val = get_labels(val_df)

    # Batch tensor so we can iterate over inputs
    train_loader = torch.utils.data.DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    n_classes = len(train_df.label.unique())
    lr = 5e-5
    eps = 1e-6
    wd = 0.01
    training_model, optimizer = get_cls_model_and_optimizer(model, n_classes, lr, eps, wd, device)

    loss_fn = torch.nn.CrossEntropyLoss()

    epochs = 3
    patience = 2
    train_on_dataset(training_model, train_loader, val_loader, optimizer, loss_fn, n_classes, epochs, patience, device)

    return model

def train_on_dataset(model, train, val, optim, loss_fn, n_classes, epochs, patience, device):

    eval_engine = create_eval_engine(model, n_classes, device)

    epochs_since_last_best = 0
    best_score = 0

    for i in range(epochs):
        # Train on all batches
        model.train()
        for batch in train:
            train_on_batch(model, batch, optim, loss_fn, n_classes, device)

        # Eval on validation set
        model.eval()
        results = eval_engine.run(val).metrics

        # Calculate whether patience has been exceeded or not
        target_score = results["average_f1"]
        if target_score > best_score:
            best_score = target_score
            epochs_since_last_best = 0
        else:
            epochs_since_last_best += 1

        if epochs_since_last_best > patience:
            break

def train_on_batch(model, batch, optim, loss_fn, n_classes, device):
    # Get the Xs and Ys
    inputs = batch[:-1]
    golds = batch[-1].to(device)

    # Put the Ys into the model
    logits = task.model(inputs)

    # Get the loss between the model outputs and the gold values
    loss = loss_fn(logits.view(-1, n_classes), golds.view(-1))

    #Backpropagate the error through the model
    loss.backward()
    optim.step()
    # Reset the gradient of the model
    model.zero_grad()
