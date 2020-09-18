import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

from mapping.mapping_models.mapping_models_base import BaseMapper

class LabseMapper(BaseMapper):

    def get_embeds(self):
        # Get text data
        df = self.get_dataset(dataset_name=self.test_dataset, app_name=self.app_name)

        # Get embeddings from text using model
        embeddings = self.get_labse_embeddings(list(df["text"]))

        # return embeddings and labels associated with those embeddings
        return embeddings.numpy(), df

    def get_mapping_name(self):
        # Use is dataset agnostic (I.e. we do not train it)
        return "labse"

    def get_labse_embeddings(self, sentence_list):

        # from sentence-transformers
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0] #First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask

        tokenizer = AutoTokenizer.from_pretrained("pvl/labse_bert", do_lower_case=False)
        model = AutoModel.from_pretrained("pvl/labse_bert")

        encoded_input = tokenizer(sentence_list, padding=True, truncation=True, max_length=256, return_tensors='pt')

        # Batch tensor so we can iterate over inputs
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(encoded_input['input_ids'],
                                                  encoded_input['token_type_ids'],
                                                  encoded_input['attention_mask']), batch_size=64, shuffle=False)

        embed_list = []

        for input_ids, token_type_ids, attention_mask in test_loader:

            with torch.no_grad():
                model_output = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

            sentence_embeddings = mean_pooling(model_output, attention_mask)

            embed_list.append(sentence_embeddings)

        return torch.cat(embed_list, dim=0)
