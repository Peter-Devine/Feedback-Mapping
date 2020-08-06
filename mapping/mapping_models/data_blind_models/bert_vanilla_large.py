from mapping.mapping_models.data_blind_models.bert_vanilla import BertVanillaMapper

class BertVanillaLargeMapper(BertVanillaMapper):

    def set_parameters(self):
        self.model_name = 'bert-large-cased'
        self.max_length = 128
        self.batch_size = 64

    def get_mapping_name(self):
        return "bert_vanilla_large"
