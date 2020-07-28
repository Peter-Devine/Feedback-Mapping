from typing import List, Tuple
import numpy as np
from nltk import wordpunct_tokenize
import requests
import os
import zipfile
import io

from mapping.mapping_models.mapping_models_base import BaseMapper

from tqdm import tqdm

# Credit https://github.com/fursovia/geometric_embedding

class GemMapper(BaseMapper):
    def get_embeds(self):
        df = self.get_dataset(self.test_dataset, split="test")

        embedding_matrix, vocab = self.download_and_get_matrix(embedding_size = 300)

        embedder = SentenceEmbedder(df.text, embedding_matrix, vocab)
        embeddings = embedder.gem()
        return embeddings, df

    def download_and_get_matrix(self, embedding_size):
        glove_dir = os.path.join(self.model_repo_dir, "glove")
        glove_saved_file = os.path.join(glove_dir, f"glove.6B.{embedding_size}d.txt")
        if not os.path.exists(glove_saved_file):
            print("Downloading Glove for GEM")
            r = requests.get("http://nlp.stanford.edu/data/glove.6B.zip")
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(path = glove_dir)

        embedding_matrix, vocab = get_embedding_matrix(glove_saved_file)
        return embedding_matrix, vocab

    def get_mapping_name(self):
        return "gem"

class SentenceEmbedder:

    def __init__(self,
                 sentences_raw: List[str],
                 embedding_matrix: np.ndarray,
                 vocab: dict) -> None:

        self.vocab = vocab
        self.sentences_raw = sentences_raw
        self.sentences = []

        for sent in self.sentences_raw:
            self.sentences.append(sentence_to_indexes(sent, self.vocab))

        self.embedding_matrix = embedding_matrix
        self.emb_dim = self.embedding_matrix.shape[1]
        self.singular_values = None

    def gem(self,
            window_size: int = 7,
            k: int = 45,
            h: int = 17,
            sigma_power: int = 3,
            ngrams: int = 1) -> np.ndarray:
        """
        Runs GEM algorithm. See the paper for detailed explanation of arguments.
        Args:
            window_size: int, the size of the window
            k: int, number of singular values
            h: int, number of principal vectors
            sigma_power: int, power for the function in (7)
            ngrams: int, n-grams size
        Returns:
            sentence_embeddings: shape [n, d]
        """
        X = np.zeros((self.emb_dim, len(self.sentences)))
        embedded_sentences = []

        print("Running GEM part 1")
        for i, sent in tqdm(enumerate(self.sentences)):
            embedded_sent = inds_to_embeddings(sent, self.embedding_matrix, ngrams)
            embedded_sentences.append(embedded_sent)
            U, s, Vh = np.linalg.svd(embedded_sent, full_matrices=False)
            X[:, i] = U.dot(s ** sigma_power)

        D, s, _ = np.linalg.svd(X, full_matrices=False)
        self.singular_values = s.copy()
        D = D[:, :k]
        s = s[:k]

        C = np.zeros((self.emb_dim, len(self.sentences)))

        print("Running GEM part 2")
        for j, sent in tqdm(enumerate(self.sentences)):
            embedded_sent = embedded_sentences[j]
            order = s * np.linalg.norm(embedded_sent.T.dot(D), axis=0)
            toph = order.argsort()[::-1][:h]
            alpha = np.zeros(embedded_sent.shape[1])

            for i in range(embedded_sent.shape[1]):
                window_matrix = self._context_window(i, window_size, embedded_sent)
                Q, R = modified_gram_schmidt_qr(window_matrix)
                q = Q[:, -1]
                r = R[:, -1]
                alpha_n = np.exp(r[-1] / (np.linalg.norm(r, ord=2, axis=0)) + 1e-18)
                alpha_s = r[-1] / window_matrix.shape[1]
                alpha_u = np.exp(-np.linalg.norm(s[toph] * (q.T.dot(D[:, toph]))) / h)
                alpha[i] = alpha_n + alpha_s + alpha_u

            C[:, j] = embedded_sent.dot(alpha)
            C[:, j] = C[:, j] - D.dot(D.T.dot(C[:, j]))

        sentence_embeddings = C.T
        return sentence_embeddings

    def mean_embeddings(self) -> np.ndarray:
        """
        Averages embeddings of words to get a sentence representation.
        Returns:
            sentence_embeddings: shape [n, d]
        """
        C = np.zeros((self.emb_dim, len(self.sentences)))

        for i, sent in enumerate(self.sentences):
            embedded_sent = inds_to_embeddings(indexes=sent, emb_matrix=self.embedding_matrix)
            C[:, i] = np.mean(embedded_sent, axis=1)

        sentence_embeddings = C.T
        return sentence_embeddings

    def _context_window(self, i: int, m: int, embeddings: np.ndarray) -> np.ndarray:
        """
        Given embedded sentence returns  the contextual window matrix of word w_i
        """
        left_window = embeddings[:, i - m:i]
        right_window = embeddings[:, i + 1:i + m + 1]
        word_embedding = embeddings[:, i][:, None]
        window_matrix = np.hstack([left_window, right_window, word_embedding])
        return window_matrix

def modified_gram_schmidt_qr(A):
    nrows, ncols = A.shape
    Q = np.zeros((nrows, ncols))
    R = np.zeros((ncols, ncols))
    for j in range(ncols):
        u = np.copy(A[:, j])
        for i in range(j):
            proj = np.dot(u, Q[:, i]) * Q[:, i]
            u -= proj

        u_norm = np.linalg.norm(u, ord=2, axis=0)
        if u_norm != 0:
            u /= u_norm
        Q[:, j] = u

    for j in range(ncols):
        for i in range(j + 1):
            R[i, j] = A[:, j].dot(Q[:, i])

    return Q, R

def preprocess_sentence(sentence: str):
    """
    :return: list of words
    """
    tokens = list(filter(str.isalpha, wordpunct_tokenize(sentence.lower())))
    return tokens

def get_embedding_matrix(path: str, skip_line: bool = False) -> Tuple[np.ndarray, dict]:
    """
    Function returns:
    1) Embedding matrix
    2) Vocabulary
    """

    embeddings = dict()
    vocabulary = []

    print("Preparing GEM")

    with open(path, 'r', encoding="utf-8") as file:
        if skip_line:
            file.readline()
        for line in file:
            values = line.split()
            word = values[0]
            coefs = np.array(values[1:], dtype=np.float64)
            embeddings[word] = coefs
            vocabulary.append(word)

    embedding_size = list(embeddings.values())[1].shape[0]

    embedding_matrix = np.zeros((len(vocabulary) + 1, embedding_size))
    embedding_matrix[-1] = np.mean(np.array(list(embeddings.values())), axis=0)

    vocab = dict()
    vocab['UNKNOWN_TOKEN'] = len(vocabulary)
    for i, word in enumerate(vocabulary):
        embedding_matrix[i] = embeddings[word]
        vocab[word] = i

    return embedding_matrix, vocab


def tokens_to_indexes(words: List[str], vocab: dict) -> List[int]:
    indexes = []
    for word in words:
        if word in vocab:
            indexes.append(vocab[word])
    return indexes


def sentence_to_indexes(sentence: str, vocab: dict) -> List[int]:
    tokens = preprocess_sentence(sentence)
    if len(tokens) == 0:
        return [vocab['UNKNOWN_TOKEN']]
    indexes = []
    for token in tokens:

        if token in vocab:
            indexes.append(vocab[token])
        else:
            indexes.append(vocab['UNKNOWN_TOKEN'])

    return indexes


def inds_to_embeddings(indexes: List[int], emb_matrix: np.ndarray, ngrams: int = 1) -> np.ndarray:
    if ngrams > 1:
        embedded_sent = emb_matrix[indexes]
        sent_len = len(indexes)
        remainder = sent_len % ngrams

        splitted = np.split(embedded_sent, np.arange(ngrams, sent_len, ngrams))

        if remainder == 0:
            embedded_sent = np.mean(splitted, axis=1)
        else:
            padded = np.zeros(splitted[0].shape)
            padded[:remainder] = splitted[-1]
            splitted[-1] = padded
            embedded_sent = np.mean(splitted, axis=1)

        return embedded_sent.T
    # shape: [d, n] (embedding dim, number of words)
    return emb_matrix[indexes].T
