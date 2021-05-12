import csv
import sys
import string
import re
import os

import torch
from torch import nn
from torch import tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

def words2charindices(sents):
    """ Convert list of sentences of words into list of list of list of character indices.
    @param sents (list[list[str]]): sentence(s) in words
    @return word_ids (list[list[list[int]]]): sentence(s) in indices
    """
    results = []
    for s in sents:
        y = []
        for w in s:
            x = [1]
            for c in list(w):
                if c in char2id:
                    x.append(char2id[c])
                else:
                    x.append(char2id['<unk>'])
            x.append(2)
            y.append(x)
        results.append(y)

    return results

def to_input_tensor_char(sents, device,max_sent_len):
    """ Convert list of sentences (words) into tensor with necessary padding for
    shorter sentences.

    @param sents (List[List[str]]): list of sentences (words)
    @param device: device on which to load the tensor, i.e. CPU or GPU

    @returns sents_var: tensor of (max_sentence_length, batch_size, max_word_length)
    """
    char_ids = words2charindices(sents)
    chars_t = pad_sents_char(char_ids, char2id['<pad>'],max_sent_len)
    chars_var = torch.tensor(chars_t, dtype=torch.long,device=device)
    chars_var = chars_var.permute(1,0,2)
    chars_var = chars_var.contiguous()

    return chars_var
    

def pad_sents_char(sents, char_pad_token,max_sent_length=150):
    """ Pad list of sentences according to the longest sentence in the batch and max_word_length.
    @param sents (list[list[list[int]]]): list of sentences, result of `words2charindices()`
        from `vocab.py`
    @param char_pad_token (int): index of the character-padding token
    @returns sents_padded (list[list[list[int]]]): list of sentences where sentences/words shorter
        than the max length sentence/word are padded out with the appropriate pad token, such that
        each sentence in the batch now has same number of words and each word has an equal
        number of characters
        Output shape: (batch_size, max_sentence_length, max_word_length)
    """
    max_word_length = 21
    sents_padded = []

    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    for s in sents:
        s_padded = []
        for w in s:
            w_padded = []
            for c in w:
                w_padded.append(c)
                if len(w_padded) >= max_word_length:
                    break
            w_diff = max_word_length - len(w_padded)
            for i in range(w_diff):
                w_padded.append(char_pad_token)
            s_padded.append(w_padded)
            if len(s_padded) >= max_sent_length:
                break
        s_diff = max_sent_length - len(s_padded)
        for i in range(s_diff):
            s_padded.append( max_word_length*[char_pad_token] )
        sents_padded.append(s_padded)
    return sents_padded