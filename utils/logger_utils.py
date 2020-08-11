import os

from utils.utils import create_dir

class TrainingLogger:
    def __init__(self, test_dataset):
        self.log_dir = os.path.join(".", "mapping", "model_training", "training_logs")
        create_dir(self.log_dir)
