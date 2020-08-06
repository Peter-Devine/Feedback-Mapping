import pandas as pd

from mapping.mapping_models.data_fit_models.nsp_lm.bert_nsp_trained_mtl import BertNspTrainedMtlMapper

class BertNspTrainedOutsideMapper(BertNspTrainedMtlMapper):

    def get_model_name(self):
        return "outside.pt"

    def get_training_data(self):
        # Use the review dataset gathered for https://giograno.me/assets/pdf/workshop/wama17.pdf
        df = pd.read_csv("https://raw.githubusercontent.com/sealuzh/user_quality/master/csv_files/reviews.csv", index_col=0)
        df = df.rename({"review": "text", "star": "sublabel"})

        return df

    def get_mapping_name(self):
        return f"bert_nsp_trained_outside"
