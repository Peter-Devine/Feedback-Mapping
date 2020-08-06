import pandas as pd

from mapping.mapping_models.data_fit_models.metadata_lm.bert_cls_trained_sublabel_mtl import BertClsTrainedSublabelMtlMapper

class BertClsTrainedSublabelOutsideMapper(BertClsTrainedSublabelMtlMapper):

    def get_model_name(self):
        return "outside.pt"

    def get_training_data(self):
        # Use the review dataset gathered for https://giograno.me/assets/pdf/workshop/wama17.pdf
        df = pd.read_csv("https://raw.githubusercontent.com/sealuzh/user_quality/master/csv_files/reviews.csv", index_col=0)
        df = df.rename({"review": "text", "star": "sublabel"})

        return {"outside_dataset": {"MISC_APP": df}}

    def get_mapping_name(self):
        return f"bert_cls_trained_sublabel_outside"
