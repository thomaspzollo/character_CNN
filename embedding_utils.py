import sys
import string
import os

import numpy as np
from sklearn.decomposition import TruncatedSVD

def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): list of distinct words across the corpus, sorted (using python 'sorted' function)
            num_corpus_words (integer): number of distinct words across the corpus
    """
    corpus_words = []
    num_corpus_words = -1
    flat_words = [ item for elem in corpus for item in elem]
    corpus_words = sorted(list(dict.fromkeys(flat_words)))
    num_corpus_words = len(corpus_words)
    return corpus_words, num_corpus_words

def compute_embedding_matrix(corpus, window_size=4):

    words, num_words = distinct_words(corpus)
    M = np.zeros((len(corpus),num_words))
    word2Ind = dict(zip(words,range(len(words))))

    for i,doc in enumerate(corpus):
        for j, word in enumerate(doc):
            word_idx = word2Ind[word]
            M[i,word_idx] += 1

    return M, word2Ind

def reduce_to_k_dim(M, k=2,cutoff=None):

    np.random.seed(4355)
    n_iters = 10     # Use this parameter in your call to `TruncatedSVD`
    M_reduced = None
    print("Running Truncated SVD over %i lyrics..." % (M.shape[0]))
    
    # ### START CODE HERE ###
    svd = TruncatedSVD(n_components=k, n_iter=n_iters)
    if cutoff is None:
        svd.fit(M)
    else:
        print("Fit using only first " + str(cutoff) + " (training) examples")
        svd.fit(M[:cutoff,:])
    M_reduced = svd.transform(M)

    # ### END CODE HERE ###

    print("Done.")
    return M_reduced