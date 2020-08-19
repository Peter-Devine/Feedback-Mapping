from mapping.mapping_models.data_blind_models.bert_likes.bert_vanilla import BertVanillaMapper

class DeclutrVanillaMapper(BertVanillaMapper):

    def set_parameters(self):
        self.model_name = 'johngiorgi/declutr-base'
        self.max_length = 128
        self.eval_batch_size = 256

    def get_mapping_name(self):
        return "declutr"
