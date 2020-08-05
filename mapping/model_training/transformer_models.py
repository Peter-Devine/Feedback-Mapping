from transformers import AutoModel,  AdamW, AutoModelForMaskedLM
from torch import nn
import torch

# Returns an optimizer that is fit to the supplied model, with the given learning rate, epsilon and weight decay parameters.
def get_weighted_adam_optimizer(model, params):
    lr = params["lr"]
    eps = params["eps"]
    wd = params["wd"]

    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
          {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": wd,
          },
          {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0},
      ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)

    return optimizer

# Returns a model which classifies one piece of text using the supplied language model and predicts over n_classes
def get_cls_model_and_optimizer(language_model, n_classes, params, device):

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

    optimizer = get_weighted_adam_optimizer(cls_lm, params)

    return cls_lm, optimizer

# Returns a model which embeds two pieces of text, and then finds the cosine similarity of these embeddings.
# This returns a binary similarity score for each pair of text.
def get_nsp_model_and_optimizer(language_model, params, device):

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
          sim = self.cos(x1[0].mean(dim=1), x2[0].mean(dim=1))

          sim = sim.view(sim.shape[0], 1)

          sim = (sim + 1) / 2
          sim_complement = 1 - sim

          logits = torch.cat((sim_complement, sim), dim=1)

          return logits

    nsp_lm = NextSentenceLanguageModel(language_model)

    nsp_lm = nsp_lm.to(device)

    optimizer = get_weighted_adam_optimizer(nsp_lm, params)

    return nsp_lm, optimizer

# Returns a model which is trained on masking words in data
def get_masking_model_and_optimizer(params, device):

    model_name = params["model_name"]

    masking_model = AutoModelForMaskedLM.from_pretrained(model_name)

    masking_model = masking_model.to(device)

    optimizer = get_weighted_adam_optimizer(masking_model, params)

    return masking_model, optimizer
