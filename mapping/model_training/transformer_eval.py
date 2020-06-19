from ignite.engine import Engine
from ignite.metrics import Accuracy, Recall, Precision, ConfusionMatrix
from ignite.metrics import MetricsLambda
import torch

def create_eval_engine(model, n_classes, device):

    process_function = get_process_function(model, device)

    eval_engine = Engine(process_function)

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

    return eval_engine

def get_process_function(model, device):
    def process_function(engine, batch):

        with torch.no_grad():
            X = batch[:-1]
            y = batch[-1]

            pred = model(X)
            gold = y.to(device)

        return pred, gold
    return process_function
