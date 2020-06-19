from transformers import AutoModel,  AdamW
from torch import nn
import torch

def get_cls_model_and_optimizer(language_model, n_classes, lr, eps, wd, device):

    class ClassificationLanguageModel(nn.Module):
      def __init__(self, lang_model, n_classes):
          super(ClassificationLanguageModel, self).__init__()
          self.lang_model = lang_model
          self.n_classes = n_classes
          hidden_size = language_model.config.hidden_size if "hidden_size" in vars(language_model.config).keys() else language_model.config.dim
          self.linear = nn.Linear(hidden_size, n_classes)

      def forward(self, x):
          x = self.lang_model(input_ids=x[0].to(device))
          x = self.linear(x[0].mean(dim=1))
          return x

    cls_lm = ClassificationLanguageModel(language_model, n_classes)

    cls_lm = cls_lm.to(device)

    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
          {
                "params": [p for n, p in cls_lm.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": wd,
          },
          {
                "params": [p for n, p in cls_lm.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0},
      ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)

    return cls_lm, optimizer

def get_nsp_model_and_optimizer(language_model, lr, eps, wd, device):

    class NextSentenceLanguageModel(nn.Module):
      def __init__(self, lang_model):
          super(NextSentenceLanguageModel, self).__init__()
          self.lang_model = lang_model
          hidden_size = language_model.config.hidden_size if "hidden_size" in vars(language_model.config).keys() else language_model.config.dim
          self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

      def forward(self, x):
          x1, x2 = x
          x1 = self.lang_model(input_ids=x1.to(device))
          x2 = self.lang_model(input_ids=x2.to(device))
          sim = self.cos(x1, x2)

          sim = (sim + 1) / 2
          sim_complement = 1 - sim

          logits = torch.cat((sim_complement, sim), dim=1)

          return logits

    cls_lm = NextSentenceLanguageModel(language_model)

    cls_lm = cls_lm.to(device)

    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
          {
                "params": [p for n, p in cls_lm.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": wd,
          },
          {
                "params": [p for n, p in cls_lm.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0},
      ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)

    return cls_lm, optimizer
