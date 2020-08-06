from mapping.mapping_models.data_blind_models.bert_vanilla import BertVanillaMapper

class AlbertVanillaMapper(BertVanillaMapper):

    def set_parameters(self):
        self.model_name = 'albert-xxlarge-v2'
        self.max_length = 128
        self.batch_size = 64

    def get_mapping_name(self):
        return "albert_vanilla"
