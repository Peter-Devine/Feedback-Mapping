import numpy as np
from numpy import linalg as LA
from scipy.stats import pearsonr
import nltk
import io
import random
from tqdm import tqdm
import os
import requests

from mapping.mapping_models.mapping_models_base import BaseMapper

# Credit https://github.com/ziyi-yang/GEM

class GemMapper(BaseMapper):

    def get_embeds(self):
        df = self.get_dataset(self.test_dataset, split="test")

        self.prepare_data()

        embeddings = self.encoder(df.text, df.text)

        return embeddings, df.label

    def get_mapping_name(self):
        return "gem-original"

    def prepare_data(self):
        self.EPS = 5e-7

        self.emb_matrix = self.download_file_and_load("1g638et14zO2bcg-acPjlkZFAk6ldAfQV", "emb_{0}.npy".format("lexvec"))
        self.word2id = self.download_file_and_load("1mc7lBmX5gucu9fRQanRTBmcNq-87qYwW", "word2id_{0}.npy".format("lexvec"))
        self.word2id = self.word2id.item()

        self.emb_matrix_psl = self.download_file_and_load("1SpXfz984Fg6MlaYQHnJYtfteS5A4fyO4", "emb_{0}.npy".format("psl"))
        self.word2id_psl = self.download_file_and_load("1pQajKlu3iLN_H9oJMBvoZSmwVVSFPUPC", "word2id_{0}.npy".format("psl"))
        self.word2id_psl = self.word2id_psl.item()

        self.emb_matrix_ftt = self.download_file_and_load("1MEia0SXgVMUocrTOfm09lk8x7AJDC4Xv", "emb_{0}.npy".format("ftt"))
        self.word2id_ftt = self.download_file_and_load("1P0SebhvYeToFr63pJcMkVvUPuaEcSb_c", "word2id_{0}.npy".format("ftt"))
        self.word2id_ftt = self.word2id_ftt.item()

        self.oov = {}
        self.oov_psl = {}
        self.oov_ftt = {}

    def download_file_and_load(self, id, file_name):
        URL = "https://docs.google.com/uc?export=download"

        destination = os.path.join(self.model_dir, file_name)

        if not os.path.exists(destination):
            session = requests.Session()

            response = session.get(URL, params = { 'id' : id }, stream = True)
            token = self.get_confirm_token(response)

            if token:
                params = { 'id' : id, 'confirm' : token }
                response = session.get(URL, params = params, stream = True)

            self.save_response_content(response, destination)

        np_file = np.load(destination, allow_pickle=True, encoding = 'latin1')

        return np_file

    def get_confirm_token(self, response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(self, response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            print(f"Downloading {destination}")
            for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    def gs(self, A):
        """
        Applies the Gram-Schmidt method to A
        and returns Q and R, so Q*R = A.
        """
        R = np.zeros((A.shape[1], A.shape[1]))
        Q = np.zeros(A.shape)
        A_c = np.copy(A)
        for k in range(0, A.shape[1]):
            R[k, k] = np.sqrt(np.dot(A_c[:, k], A_c[:, k]))
            if R[k, k] < self.EPS:
                R[k, k] = 0
                continue
            Q[:, k] = A_c[:, k]/R[k, k]
            for j in range(k+1, A.shape[1]):
                R[k, j] = np.dot(Q[:, k], A_c[:, j])
                A_c[:, j] = A_c[:, j] - R[k, j]*Q[:, k]
        return Q, R

    def sent_to_tokens(self, sent):
        sent = sent.replace("''", '" ')
        sent = sent.replace("``", '" ')
        tokens = [token.lower().replace("``", '"').replace("''", '"') for token in nltk.wordpunct_tokenize(sent)]
        return tokens

    def rm_pr(self, m, C_0):
        if C_0.ndim == 1:
            C_0 = np.reshape(C_0, [-1, 1])

        w = np.transpose(C_0).dot(m)
        return m - C_0.dot(w)

    def ngram(self, s_num, C_0, sgv_c, win_sz = 7):
        n_pc = np.shape(C_0)[1]
        num_words = np.shape(s_num)[1]
        wgt = np.zeros(num_words)

        for i in range(num_words):
            beg_id = max(i - win_sz, 0)
            end_id = min(i + win_sz, num_words - 1)
            ctx_ids = list(range(beg_id, i)) + list(range(i+1, end_id + 1))
            m_svd = np.concatenate((s_num[:, ctx_ids], (s_num[:, i])[:, np.newaxis]), axis = 1)

            U, sgv, _ = LA.svd(m_svd, full_matrices = False)

            l_win = np.shape(U)[1]
            q, r = self.gs(m_svd)
            norm = LA.norm(s_num[:, i], 2)

            w = q[:, -1].dot(U)
            w_sum = LA.norm(w*sgv, 2)/l_win

            kk = sgv_c*(q[:, -1].dot(C_0))
            wgt[i] = np.exp(r[-1, -1]/norm) + w_sum + np.exp((-LA.norm(kk, 2))/n_pc)
        # print wgt
        return wgt

    def sent_to_ids(self, sent, word2id, tokens, oov):
        """
        sent is a string of chars, return a list of word ids
        """
        if tokens is None:
            tokens = self.sent_to_tokens(sent)
        ids = []

        for w in tokens:
            if w in ['!', '.', ':', '?', '@', '-', '"', "'"]: continue
            if w in word2id:
                id = word2id[w]
            elif 'unk' in word2id:
                # OOV tricks
                if w in oov:
                    id = oov[w]
                else:
                    id = random.choice(range(len(word2id)))
                    oov[w] = id
            ids.append(id)
        return ids

    def str_2_num(self, s1):
        tokens = self.sent_to_tokens(s1)
        s_num1 = self.emb_matrix[self.sent_to_ids(s1, self.word2id, tokens, self.oov), :]
        s_num2 = self.emb_matrix_psl[self.sent_to_ids(s1, self.word2id_psl, tokens, self.oov_psl), :]
        s_num3 = self.emb_matrix_ftt[self.sent_to_ids(s1, self.word2id_ftt, tokens, self.oov_ftt), :]
        matrix = np.transpose(np.concatenate((s_num1, s_num2, s_num3), axis = 1))
        return matrix

    def svd_sv(self, s1, factor = 3):
        s_num = self.str_2_num(s1)
        U, s, Vh = LA.svd(s_num, full_matrices = False)
        vc = U.dot(s**factor)
        return vc

    def feat_extract(self, m1, n_rm, C_all, soc):
        w1 = LA.norm(np.transpose(m1).dot(C_all)*soc, axis = 0)
        id1 = w1.argsort()[-n_rm:]
        return id1

    def encoder(self, encoding_list, corpus_list, dim = 900, n_rm = 17, max_n = 45, win_sz = 7):
        """
        corpus_list: the list of corpus, in the case of STS benchmark, it's s1 + s2
        encoding_list: the list of sentences to encode
        dim: the dimension of sentence vector
        """
        s_univ = np.zeros((dim, len(corpus_list)))
        encoded = []
        print("Creating GEM corpus list...")
        for j, sent in tqdm(enumerate(corpus_list)):
            s_univ[:, j] = self.svd_sv(sent)
        U, s, V = LA.svd(s_univ, full_matrices = False)
        C_all = U[:, :max_n]
        soc = s[:max_n]
        print("Getting GEM embeddings...")
        for j, sent in tqdm(enumerate(encoding_list)):
            m = self.str_2_num(sent)
            id1 = self.feat_extract(m, n_rm, C_all, soc)
            C_1 = C_all[:, id1]
            sgv = soc[id1]
            m_rm = self.rm_pr(m, C_1)
            v = m_rm.dot(self.ngram(m, C_1, sgv, win_sz))
            encoded.append(v)
        encoded = np.asarray(encoded)
        return encoded
