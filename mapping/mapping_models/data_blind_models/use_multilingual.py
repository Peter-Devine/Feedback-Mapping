import os
import requests
import zipfile
import tarfile

import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
from mapping.mapping_models.mapping_models_base import BaseMapper

class UseMultilingualMapper(BaseMapper):

    def get_embeds(self):
        # Get text data
        df = self.get_dataset(self.test_dataset, split="test")

        # Get model
        embed = self.get_model()

        # Get embeddings from text using model
        embeddings = embed(df["text"])

        # return embeddings and labels associated with those embeddings
        return embeddings.numpy(), df

    def get_model(self):
        # Check to see if a USE model file has already been downloaded.
        # If so, use the downlaoded version.
        # If not, download and then save it before using.
        model_file = os.path.join(self.model_dir, "use_tf")
        if not os.path.exists(model_file):
            # Create folder to save model data into
            os.mkdir(model_file)

            # Download file data to .tar.gz file
            r = requests.get("https://storage.googleapis.com/tfhub-modules/google/universal-sentence-encoder-multilingual-large/3.tar.gz", stream=True)
            zip_file = os.path.join(model_file, "3.tar.gz")

            # Save zipped model data to disk
            print("Streaming USE multilingual data now")
            with open(zip_file, 'wb') as f:
                f.write(r.raw.read())

            # Unzip model data
            print("Unzipping USE multilingual data now")
            tar = tarfile.open(zip_file, "r:gz")
            tar.extractall(path = model_file)
            tar.close()

            # Delete model zip data
            print("Removing USE multilingual zip file now")
            os.remove(zip_file)

        # Load model from disc
        embed = hub.load(model_file)

        return embed

    def get_mapping_name(self):
        # Use is dataset agnostic (I.e. we do not train it)
        return "use_multilingual"
