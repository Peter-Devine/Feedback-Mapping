import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from tqdm import tqdm

import torch

from transformers import AutoConfig, AutoTokenizer, AutoModelWithLMHead

from mapping.mapping_models.mapping_models_base import BaseMapper

# Credit https://github.com/BinWang28/SBERT-WK-Sentence-Embedding

class SBertWKMapper(BaseMapper):

    def get_embeds(self):
        df = self.get_dataset(dataset_name=self.test_dataset, app_name=self.app_name)

        # Get embeddings from text series
        embeddings = self.get_sentence_embeds(df.text.values)

        return embeddings, df

    def get_sentence_embeds(self, sentences):
        self.model_name = "binwang/bert-base-nli"
        self.eval_batch_size = 256

        #Get SBERT-WK model and tokenizer
        config = AutoConfig.from_pretrained(self.model_name, cache_dir=self.model_dir)
        config.output_hidden_states = True
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_dir)
        model = AutoModelWithLMHead.from_pretrained(self.model_name, config=config, cache_dir=self.model_dir)
        model = model.to(self.device)
        model.eval()
        model.zero_grad()

        params = {
            "max_seq_length": 128,
            "context_window_size": 2,
            "layer_start": 4
        }

        test_loader = torch.utils.data.DataLoader(sentences, batch_size=self.eval_batch_size, shuffle=False)

        embeddings = []
        print("Inferencing on SBERT-WK")
        for sentence_batch in tqdm(test_loader):
            inputs, features_mask = self.get_model_inputs(sentence_batch, tokenizer, params)

            with torch.no_grad():
                features = model(**inputs)[1]

            # Reshape features from list of (batch_size, seq_len, hidden_dim) for each hidden state to list
            # of (num_hidden_states, seq_len, hidden_dim) for each element in the batch.
            all_layer_embedding = torch.stack(features).permute(1, 0, 2, 3).cpu().numpy()

            embedding = self.dissecting(params, all_layer_embedding, features_mask)
            embeddings.append(embedding)

        embeddings = np.concatenate(embeddings, axis=0)

        return embeddings

    def get_model_inputs(self, sentences, tokenizer, params):
        sentences_index = [tokenizer.encode(s, add_special_tokens=True) for s in sentences]
        features_input_ids = []
        features_mask = []
        for sent_ids in sentences_index:
            # Truncate if too long
            if len(sent_ids) > params["max_seq_length"]:
                sent_ids = sent_ids[: params["max_seq_length"]]
            sent_mask = [1] * len(sent_ids)
            # Padding
            padding_length = params["max_seq_length"] - len(sent_ids)
            sent_ids += [0] * padding_length
            sent_mask += [0] * padding_length
            # Length Check
            assert len(sent_ids) == params["max_seq_length"]
            assert len(sent_mask) == params["max_seq_length"]

            features_input_ids.append(sent_ids)
            features_mask.append(sent_mask)

        features_mask = np.array(features_mask)

        batch_input_ids = torch.tensor(features_input_ids, dtype=torch.long)
        batch_input_mask = torch.tensor(features_mask, dtype=torch.long)
        batch = [batch_input_ids.to(self.device), batch_input_mask.to(self.device)]

        inputs = {"input_ids": batch[0], "attention_mask": batch[1]}

        return inputs, features_mask

    def dissecting(self, params, all_layer_embedding, features_mask):
        """
            dissecting deep contextualized model
        """
        unmask_num = np.sum(features_mask, axis=1) - 1 # Not considering the last item
        all_layer_embedding = np.array(all_layer_embedding)[:,params['layer_start']:,:,:] # Start from 4th layers output

        embedding = []
        # One sentence at a time
        for sent_index in range(len(unmask_num)):
            sentence_feature = all_layer_embedding[sent_index,:,:unmask_num[sent_index],:]
            one_sentence_embedding = []
            # Process each token
            for token_index in range(sentence_feature.shape[1]):
                token_feature = sentence_feature[:,token_index,:]
                # 'Unified Word Representation'
                token_embedding = self.unify_token(params, token_feature)
                one_sentence_embedding.append(token_embedding)

            one_sentence_embedding = np.array(one_sentence_embedding)
            sentence_embedding = self.unify_sentence(params, sentence_feature, one_sentence_embedding)
            embedding.append(sentence_embedding)

        embedding = np.array(embedding)

        return embedding

    def unify_token(self, params, token_feature):
        """
            Unify Token Representation
        """
        window_size = params['context_window_size']

        alpha_alignment = np.zeros(token_feature.shape[0])
        alpha_novelty = np.zeros(token_feature.shape[0])

        for k in range(token_feature.shape[0]):

            left_window = token_feature[k-window_size:k,:]
            right_window = token_feature[k+1:k+window_size+1,:]
            window_matrix = np.vstack([left_window, right_window, token_feature[k,:][None,:]])

            Q, R = np.linalg.qr(window_matrix.T) # This gives negative weights

            q = Q[:, -1]
            r = R[:, -1]
            alpha_alignment[k] = np.mean(normalize(R[:-1,:-1],axis=0),axis=1).dot(R[:-1,-1]) / (np.linalg.norm(r[:-1]))
            alpha_alignment[k] = 1/(alpha_alignment[k]*window_matrix.shape[0]*2)
            alpha_novelty[k] = abs(r[-1]) / (np.linalg.norm(r))

        # Sum Norm
        alpha_alignment = alpha_alignment / np.sum(alpha_alignment) # Normalization Choice
        alpha_novelty = alpha_novelty / np.sum(alpha_novelty)

        alpha = alpha_novelty + alpha_alignment

        alpha = alpha / np.sum(alpha) # Normalize

        out_embedding = token_feature.T.dot(alpha)

        return out_embedding

    def unify_sentence(self, params, sentence_feature, one_sentence_embedding):
        """
            Unify Sentence By Token Importance
        """
        sent_len = one_sentence_embedding.shape[0]

        var_token = np.zeros(sent_len)
        for token_index in range(sent_len):
            token_feature = sentence_feature[:,token_index,:]
            sim_map = cosine_similarity(token_feature)
            var_token[token_index] = np.var(sim_map.diagonal(-1))

        var_token = var_token / np.sum(var_token)

        sentence_embedding = one_sentence_embedding.T.dot(var_token)

        return sentence_embedding

    def get_mapping_name(self):
        return "sbert-wk"
