from sentence_transformers import SentenceTransformer
from mapping.mapping_models.mapping_models_base import BaseMapper

class SBertMapper(BaseMapper):

    def get_embeds(self):
        df = self.get_dataset(self.test_dataset, split="test")

        # Get SBERT model
        self.model_name = 'bert-base-nli-mean-tokens'
        model = SentenceTransformer(self.model_name)
        model = model.to(self.device)

        # Get embeddings for sentences
        BATCH_SIZE = 64
        sentence_embeddings = model.encode(df.text.values, batch_size = BATCH_SIZE, show_progress_bar = True)

        return sentence_embeddings, df.label

    def get_mapping_name(self, test_dataset):
        return "sbert"
