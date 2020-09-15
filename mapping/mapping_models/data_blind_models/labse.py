# !pip install bert-for-tf2

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import bert

from mapping.mapping_models.mapping_models_base import BaseMapper

class LabseMapper(BaseMapper):

    def get_embeds(self):
        # Get text data
        df = self.get_dataset(dataset_name=self.test_dataset, app_name=self.app_name)

        # Get embeddings from text using model
        embeddings = self.get_labse_embeddings(list(df["text"]))

        # return embeddings and labels associated with those embeddings
        return embeddings.numpy(), df

    def get_mapping_name(self):
        # Use is dataset agnostic (I.e. we do not train it)
        return "labse"

    def get_labse_embeddings(self, sentence_list):

        def get_model(model_url, max_seq_length):
          labse_layer = hub.KerasLayer(model_url, trainable=True)

          # Define input.
          input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                                 name="input_word_ids")
          input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                             name="input_mask")
          segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                              name="segment_ids")

          # LaBSE layer.
          pooled_output,  _ = labse_layer([input_word_ids, input_mask, segment_ids])

          # The embedding is l2 normalized.
          pooled_output = tf.keras.layers.Lambda(
              lambda x: tf.nn.l2_normalize(x, axis=1))(pooled_output)

          # Define model.
          return tf.keras.Model(
                inputs=[input_word_ids, input_mask, segment_ids],
                outputs=pooled_output), labse_layer

        max_seq_length = 256
        labse_model, labse_layer = get_model(
            model_url="https://tfhub.dev/google/LaBSE/1", max_seq_length=max_seq_length)

        vocab_file = labse_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = labse_layer.resolved_object.do_lower_case.numpy()
        tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case)

        def create_input(input_strings, tokenizer, max_seq_length):

          input_ids_all, input_mask_all, segment_ids_all = [], [], []
          for input_string in input_strings:
            # Tokenize input.
            input_tokens = ["[CLS]"] + tokenizer.tokenize(input_string) + ["[SEP]"]
            input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
            sequence_length = min(len(input_ids), max_seq_length)

            # Padding or truncation.
            if len(input_ids) >= max_seq_length:
              input_ids = input_ids[:max_seq_length]
            else:
              input_ids = input_ids + [0] * (max_seq_length - len(input_ids))

            input_mask = [1] * sequence_length + [0] * (max_seq_length - sequence_length)

            input_ids_all.append(input_ids)
            input_mask_all.append(input_mask)
            segment_ids_all.append([0] * max_seq_length)

          return np.array(input_ids_all), np.array(input_mask_all), np.array(segment_ids_all)

        def encode(input_text):
          input_ids, input_mask, segment_ids = create_input(
            input_text, tokenizer, max_seq_length)
          return labse_model([input_ids, input_mask, segment_ids])

        batch_size = 128
        embeddings = []
        for i in range(len(sentence_list) * batch_size):
            batch_start = i * batch_size
            batch_end = (i+1) * batch_size

            batch = sentence_list[batch_start : batch_end]
            embeddings.append(encode(batch))

        embeddings = np.stack(embeddings, axis=0)

        return embeddings
