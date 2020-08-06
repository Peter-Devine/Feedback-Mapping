from sentence_transformers import SentenceTransformer
from mapping.mapping_models.mapping_models_base import BaseMapper

class SBertMapper(BaseMapper):

    def get_embeds(self):
        df = self.get_dataset(dataset_name=self.test_dataset, app_name=self.app_name)

        # Get SBERT model
        self.model_name = 'bert-base-nli-mean-tokens'
        model = SentenceTransformer(self.model_name)
        model = model.to(self.device)

        # Get embeddings for sentences
        BATCH_SIZE = 64
        sentence_embeddings = model.encode(df.text.values, batch_size = BATCH_SIZE, show_progress_bar = True)

        return sentence_embeddings, df

    def get_mapping_name(self):
        return "sbert"
