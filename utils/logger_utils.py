import os
import datetime

from utils.utils import create_dir

class TrainingLogger:
    def __init__(self):
        # First, make sure that the target dir exists
        self.log_dir = os.path.join(".", "mapping", "model_training", "training_logs")
        create_dir(self.log_dir)

        # Get a string of the current time to the second
        file_str = datetime.datetime.now().strftime("%y_%m_%d__%H_%M_%S_%f")

        # Make a file path for the new logger output
        self.file_dir = os.path.join(self.log_dir, file_str)
        assert not os.path.exists(self.file_dir), f"{self.file_dir} already exists. This probably shouldn't happen unless the TrainingLogger is initialized at the same millisecond as another logger. Please investigate."

        # We start the logging by stating the date
        readable_date = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
        self.log(f"Training log commencing at {readable_date}:\n")

    def log(self, text):
        # First we print the provided text
        print(text)

        # And then we log this same text to file for inspection afterwards
        with open(self.file_dir, "a+") as f:
            f.write(f"{text}\n")
