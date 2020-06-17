from ignite.engine import Engine
from ignite.metrics import Accuracy, Recall, Precision, ConfusionMatrix
from ignite.metrics import MetricsLambda
import torch

def create_eval_engine(model, is_multilabel, n_classes, device):

  def process_function(engine, batch):

      with torch.no_grad():
          X = batch[:-1]
          y = batch[-1]

          pred = model(X)
          gold = y.to(device)

      return pred, gold

  eval_engine = Engine(process_function)

  if is_multilabel:
      accuracy = MulticlassOverallAccuracy(n_classes=n_classes)
      accuracy.attach(eval_engine, "accuracy")
      per_class_accuracy = MulticlassPerClassAccuracy(n_classes=n_classes)
      per_class_accuracy.attach(eval_engine, "per class accuracy")
      recall = MulticlassRecall(n_classes=n_classes)
      recall.attach(eval_engine, "recall")
      precision = MulticlassPrecision(n_classes=n_classes)
      precision.attach(eval_engine, "precision")
      f1 = MulticlassF(n_classes=n_classes, f_n=1)
      f1.attach(eval_engine, "f1")
      f2= MulticlassF(n_classes=n_classes, f_n=2)
      f2.attach(eval_engine, "f2")

      avg_recall = MulticlassRecall(n_classes=n_classes, average=True)
      avg_recall.attach(eval_engine, "average recall")
      avg_precision = MulticlassPrecision(n_classes=n_classes, average=True)
      avg_precision.attach(eval_engine, "average precision")
      avg_f1 = MulticlassF(n_classes=n_classes, average=True, f_n=1)
      avg_f1.attach(eval_engine, "average f1")
      avg_f2= MulticlassF(n_classes=n_classes, average=True, f_n=2)
      avg_f2.attach(eval_engine, "average f2")
  else:
      accuracy = Accuracy()
      accuracy.attach(eval_engine, "accuracy")
      recall = Recall(average=False)
      recall.attach(eval_engine, "recall")
      precision = Precision(average=False)
      precision.attach(eval_engine, "precision")
      confusion_matrix = ConfusionMatrix(num_classes=n_classes)
      confusion_matrix.attach(eval_engine, "confusion_matrix")
      f1 = (precision * recall * 2 / (precision + recall))
      f1.attach(eval_engine, "f1")
      f2 = (precision * recall * 5 / ((4*precision) + recall))
      f2.attach(eval_engine, "f2")

      def Fbeta(r, p, beta):
          return torch.mean((1 + beta ** 2) * p * r / (beta ** 2 * p + r + 1e-20)).item()

      avg_f1 = MetricsLambda(Fbeta, recall, precision, 1)
      avg_f1.attach(eval_engine, "average f1")
      avg_f2 = MetricsLambda(Fbeta, recall, precision, 2)
      avg_f2.attach(eval_engine, "average f2")

      avg_recall = Recall(average=True)
      avg_recall.attach(eval_engine, "average recall")
      avg_precision = Precision(average=True)
      avg_precision.attach(eval_engine, "average precision")

      if n_classes == 2:
          top_k = TopK(k=10, label_idx_of_interest=0)
          top_k.attach(eval_engine, "top_k")

  return eval_engine
