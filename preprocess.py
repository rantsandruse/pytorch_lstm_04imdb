import torch
import os
import pandas as pd
import numpy as np

# From Fujita
# def load_embeddings(path):
#     with open(path) as f:
#         return dict(get_coefs(*line.strip().split(' ')) for line in tqdm(f))

def load_pretrained_embedding(filename):
    '''
    Create dictionary
    :param filename:
    :return:
    https://www.kaggle.com/hamishdickson/bidirectional-lstm-in-keras-with-glove-embeddings
    '''
    glove_embeddings_dict = {}
    f = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    for line in f:
        values = line.rstrip().split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        glove_embeddings_dict[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(glove_embeddings_dict))
    return glove_embeddings_dict

def build_embedding_matrix(embedding_dict, text_vocab, emb_dim=-1):
    '''
    :param embedding_dict: pretrained embedding dictionary
    :param text_vocab: dictionary of text vocabulary
    :param emb_dim:
    :return:
    '''
    # matrix_len: Number of works in the corpus
    matrix_len = len(text_vocab)
    # we can also arbitrary truncated the vector length.
    # emb_dim = -1 means that we will use the emb_dim from the embedding_dict.
    if emb_dim == -1: emb_dim = list(embedding_dict.values())[0]
    embedding_matrix = np.zeros((matrix_len, emb_dim))
    words_found = 0

    for i, text in enumerate(text_vocab):
        try:
            embedding_matrix[i] = embedding_dict[text][:emb_dim]
            words_found += 1
        except KeyError:
            # Using the same value
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))

    return embedding_matrix

def seq_to_embedding(seq, to_ix): # seq_length = 10
    '''
    This is a good entry point for passing in different kinds of embeddings and
    :param seq: sequence of words
    :param to_ix: embedding lib
    :return:
    '''
    idxs = [to_ix[w] for w in seq]
    # if len(idxs) < seq_length:
    #     idxs = idxs + [-1] * (seq_length - len(idxs))
    # return torch.tensor(idxs, dtype=torch.long)
    return idxs


def seqs_to_dictionary_v3(training_data: pd.DataFrame, text_col = "text"):
    word_to_ix = {'<PAD>':0}
    count1 = 1
    for sent in training_data[text_col]:
        words = sent.split()
        for word in words:
            if word not in word_to_ix:
                word_to_ix[word] = count1
                count1 +=1

    return word_to_ix


# TO DO - make it very much like v3, use a lambda function or something.
def seqs_to_dictionary_v4(reviews):
    corpus_vocab = {'<PAD>':0}
    count = 1
    for review in reviews:
        for word in review:
            if word not in corpus_vocab:
                corpus_vocab[word] = count
                count += 1

    return corpus_vocab

# This function was stolen from keras.
# I'm trying to work off pytorch alone and not leaning on tensorflow functions.
def pad_sequences(sequences,
                  maxlen=None,
                  dtype='int32',
                  padding='pre',
                  truncating='pre',
                  value=0.):
  """Pads sequences to the same length.
  This function transforms a list of
  `num_samples` sequences (lists of integers)
  into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
  `num_timesteps` is either the `maxlen` argument if provided,
  or the length of the longest sequence otherwise.
  Sequences that are shorter than `num_timesteps`
  are padded with `value` at the end.
  Sequences longer than `num_timesteps` are truncated
  so that they fit the desired length.
  The position where padding or truncation happens is determined by
  the arguments `padding` and `truncating`, respectively.
  Pre-padding is the default.
  Arguments:
      sequences: List of lists, where each element is a sequence.
      maxlen: Int, maximum length of all sequences.
      dtype: Type of the output sequences.
      padding: String, 'pre' or 'post':
          pad either before or after each sequence.
      truncating: String, 'pre' or 'post':
          remove values from sequences larger than
          `maxlen`, either at the beginning or at the end of the sequences.
      value: Float, padding value.
  Returns:
      x: Numpy array with shape `(len(sequences), maxlen)`
  Raises:
      ValueError: In case of invalid values for `truncating` or `padding`,
          or in case of invalid shape for a `sequences` entry.
  """
  if not hasattr(sequences, '__len__'):
    raise ValueError('`sequences` must be iterable.')
  lengths = []
  for x in sequences:
    if not hasattr(x, '__len__'):
      raise ValueError('`sequences` must be a list of iterables. '
                       'Found non-iterable: ' + str(x))
    lengths.append(len(x))

  num_samples = len(sequences)
  if maxlen is None:
    maxlen = np.max(lengths)

  # take the sample shape from the first non empty sequence
  # checking for consistency in the main loop below.
  sample_shape = tuple()
  for s in sequences:
    if len(s) > 0:  # pylint: disable=g-explicit-length-test
      sample_shape = np.asarray(s).shape[1:]
      break

  x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
  for idx, s in enumerate(sequences):
    if not len(s):  # pylint: disable=g-explicit-length-test
      continue  # empty list/array was found
    if truncating == 'pre':
      trunc = s[-maxlen:]  # pylint: disable=invalid-unary-operand-type
    elif truncating == 'post':
      trunc = s[:maxlen]
    else:
      raise ValueError('Truncating type "%s" not understood' % truncating)

    # check `trunc` has expected shape
    trunc = np.asarray(trunc, dtype=dtype)
    if trunc.shape[1:] != sample_shape:
      raise ValueError('Shape of sample %s of sequence at position %s '
                       'is different from expected shape %s' %
                       (trunc.shape[1:], idx, sample_shape))

    if padding == 'post':
      x[idx, :len(trunc)] = trunc
    elif padding == 'pre':
      x[idx, -len(trunc):] = trunc
    else:
      raise ValueError('Padding type "%s" not understood' % padding)
  return x






