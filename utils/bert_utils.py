import torch
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

def get_lm_embeddings(mapper_model, test_df, trained_model_name, use_first_token_only = False):
    mapper_model.set_parameters()

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = AutoTokenizer.from_pretrained(mapper_model.model_name, use_fast=True)

    # Load the language model
    model = mapper_model.get_model()
    model = model.to(mapper_model.device)
    model.eval()
    model.zero_grad()

    # Tokenize and convert to input IDs
    tokens_tensor = tokenizer.batch_encode_plus(list(test_df.text.values),
                                                max_length = mapper_model.max_length,
                                                pad_to_max_length=True,
                                                truncation=True,
                                                return_tensors="pt")
    tokens_tensor = tokens_tensor["input_ids"]

    # Create list for all embeddings to be saved to
    embeddings = []

    # Batch tensor so we can iterate over inputs
    test_loader = torch.utils.data.DataLoader(tokens_tensor, batch_size=mapper_model.eval_batch_size, shuffle=False)

    # Make sure the torch algorithm runs without gradients (as we aren't training)
    with torch.no_grad():
        print(f"Iterating over inputs {trained_model_name}")
        # Iterate over all batches, passing the batches through the test set
        for test_batch in tqdm(test_loader):
            # Get the model output from the test set
            outputs = model(test_batch.to(mapper_model.device))

            if use_first_token_only:
                # Output only the model output from the first token position (I.e. the position that BERT NSP is trained on)
                np_array = outputs[0][:,0,:].cpu().numpy()
            else:
                # Output the final average encoding across all characters as a numpy array
                np_array = outputs[0].mean(dim=1).cpu().numpy()

            # Append this encoding to a list
            embeddings.append(np_array)

    all_embeddings = np.concatenate(embeddings, axis=0)

    return all_embeddings
