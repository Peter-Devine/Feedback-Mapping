from mapping.mapping_models.data_blind_models.bert_likes.bert_vanilla import BertVanillaMapper

class SentenceTransformerBertVanillaMapper(BertVanillaMapper):

    def set_parameters(self):
        self.model_name = 'sentence-transformers/bert-base-nli-stsb-mean-tokens'
        self.max_length = 128
        self.eval_batch_size = 256

    def get_mapping_name(self):
        return "sentence_transformer_bert"
