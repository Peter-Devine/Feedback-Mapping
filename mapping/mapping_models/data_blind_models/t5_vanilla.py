from transformers import AutoModel
from mapping.mapping_models.mapping_models_base import BaseMapper
from utils.bert_utils import get_lm_embeddings

class T5VanillaMapper(BaseMapper):

    def get_embeds(self):
        test_df = self.get_dataset(dataset_name=self.test_dataset, app_name=self.app_name)

        all_embeddings = get_lm_embeddings(self, test_df, f"{self.get_mapping_name()}")

        return all_embeddings, test_df

    def get_model(self):
        model = AutoModel.from_pretrained(self.model_name)
        model = model.encoder

        return model

    def set_parameters(self):
        self.model_name = 't5-base'
        self.max_length = 128
        self.batch_size = 64

    def get_mapping_name(self):
        return "t5_vanilla"
