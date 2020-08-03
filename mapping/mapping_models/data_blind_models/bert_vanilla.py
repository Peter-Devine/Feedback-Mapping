from transformers import AutoModel
from mapping.mapping_models.mapping_models_base import BaseMapper
from utils.bert_utils import get_lm_embeddings

class BertVanillaMapper(BaseMapper):

    def get_embeds(self):
        test_df = self.get_dataset(self.test_dataset, split="test")

        all_embeddings = get_lm_embeddings(self, test_df, f"{self.get_mapping_name()}")

        return all_embeddings, test_df

    def get_model(self):
        model = AutoModel.from_pretrained(self.model_name)

        return model

    def set_parameters(self):
        self.model_name = 'bert-base-uncased'
        self.max_length = 128
        self.batch_size = 64

    def get_mapping_name(self):
        return "bert_vanilla"
