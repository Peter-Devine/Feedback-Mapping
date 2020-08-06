from mapping.mapping_models.data_blind_models.bert_likes.bert_vanilla import BertVanillaMapper
from utils.bert_utils import get_lm_embeddings

class BertVanillaFirstTokMapper(BertVanillaMapper):

    def get_embeds(self):
        test_df = self.get_dataset(self.test_dataset, split="test")

        all_embeddings = get_lm_embeddings(self, test_df, f"{self.get_mapping_name()}", use_first_token_only = True)

        return all_embeddings, test_df

    def get_mapping_name(self):
        return "bert_vanilla_first_tok"
