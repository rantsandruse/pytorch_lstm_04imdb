'''
This is LSTM Tagger - can train against arbitary targets
What do I want to build next?
What's the improvement from the prior iteration:
1. I want to initialize hidden states - the default h0,c0 are just zero vectors.
2. I will train by batch
3. I will add padding and mask
3.
1. LSTM language generator
2. LSTM-GRM combo
3.

Reference:
1. https://gist.github.com/williamFalcon/f27c7b90e34b4ba88ced042d9ef33edd
2. https://github.com/chrisvdweth/ml-toolkit/blob/master/pytorch/notebooks/minimal-example-lstm-input.ipynb
3. https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html

'''
import torch
from torch import nn
from torch.autograd import Variable

class LSTMClassifier(nn.Module):
    '''
    This is v2 of LSTM Tagger.
    The main improvements are invoking batch_first, padding/packing and
    '''
    def __init__(self, embedding_matrix, hidden_dim, output_size, batch_size=2):
        '''
        embedding_dim: Glove is 300. We are using 6 here.
        hidden_dim: can be anything, usually 32 or 64. We are using 6 here.
        vocab_size: vocabulary size includes an index for padding.
        output_size: We need to exclude the index for padding here.
        '''
        super().__init__()
        self.hidden_dim = hidden_dim
        # Note: Change No.1: set embedding_dim using previous batch
        self.embedding_dim = embedding_matrix.shape[1]
        # In this case, vocab_size is 9, embedding_dim is 6.
        # hidden dim is also called "number of lstm units"
        # Whenever padding_idx = 0, embedding = 0.
        # Note: Change No.2: set word embedding from pretrained.
        self.word_embeddings = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32), sparse=False)
        # In this case, vocab_size is 9, embedding_dim is 6.
        # Prepare our model for minibatch based training.
        self.batch_size = batch_size

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        # If you use batch_first=True, Then the input shape will be:
        # (batch_size, seq_len, hidden_dim)
        # If you use batch_first=False, Then the input shape will be:
        # (seq_len, batch_size, hidden_dim)
        # Recommend turning it out, otherwise a pain in the neck.
        # Default number of layers is 1.

        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim, batch_first=True)


        # Note: Change No.3: Add dropout
        self.dropout = nn.Dropout(p=0.3)
        # Add layer norm
        self.lnorm = torch.nn.LayerNorm(hidden_dim, eps=1e-05, elementwise_affine=True)
        # The linear layer that maps from hidden state space to tag space
        # output_size = tagset_size - 1 to discount padding tag.
        self.hidden2tag = nn.Linear(hidden_dim, output_size)

    def init_hidden(self):
        '''
        Initiate hidden states.
        '''
        # Shape for hidden state and cell state: num_layers * num_directions, batch, hidden_size
        h_0 = torch.randn(1, self.batch_size, self.hidden_dim)
        c_0 = torch.randn(1, self.batch_size, self.hidden_dim)

        h_0 = h_0.cuda()
        c_0 = c_0.cuda()
        # The Variable API is now semi-deprecated, so we use nn.Parameter instead.
        # Note: For Variable API requires_grad=False by default;
        # For Parameter API requires_grad=True by default.

        h_0 = Variable(h_0)
        c_0 = Variable(c_0)

        #h_0 = nn.Parameter(h_0, requires_grad=False)
        #c_0 = nn.Parameter(c_0, requires_grad=False)

        return (h_0, c_0)

    def forward(self, sentences, X_lengths):
        '''
        Parameters
        ----------
        sentences: padded sentences tensor. Each element of the tensor is an array of words.
        X_lengths: length of sentence tensor. Each element of the tensor is the original
        length of the unpadded sentence.

        Returns
        -------
        '''
        # Dimensions of tensors:
        # (Note that seq_len is max length)
        # Shape of embedding (embeds): batch_size, seq_len, hidden_dim
        # Shape of embedding post packing (embeds): batch_size, orig_len, hidden_dim
        # Shape of self.hidden: (num_layers*num_directions, batch_size, hidden_dim)
        # Shape of lstm_out: batch_size, seq_len, hidden_dim
        # Shape of tag_scores: batch_size, 1
        hidden_0 = self.init_hidden()
        batch_size, seq_len = sentences.size()
        embeds = self.word_embeddings(sentences)

        # We need to reshape the batch from
        # This is the shape wanted according to lstm user manual (Need to understand why?)
        # print("embedding shape:", embeds.shape)

        # By setting batch_first=true, we are outputting a tensor of:
        # (batch_size, seq_len, input_size)
        # instead of the default:
        # (seq_len, batch_size, input_size)
        # Having batch_size first is more intuitive to human, while having seq_len as the first
        # dimension makes tensor operations easier.
        embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, X_lengths, batch_first=True, \
                                                         enforce_sorted=False)
        # Note: we no longer need to reshape the input: As we used batch first, the input to LSTM
        # here is already (batch_size, seq_len, hidden_dim)
        # The original code is:
        # lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # We choose not to save self.hidden as we re-initialize the hidden state for the new batch
        # to random anyways.
        # Note: the commented line below is what we will do if we want to train our own initial
        # state.
        # lstm_out, _ = self.lstm(embeds, (self.h_0, self.c_0))
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(embeds, hidden_0)

        # Note: parsing in total_length is a must, otherwise you might run into dimension mismatch.
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, \
                                                             total_length=seq_len)

        drop_out = self.dropout(lstm_out)
        drop_out = self.lnorm(drop_out)
        # We need to get the last non-padding feature of the sequence.
        # https://stackoverflow.com/questions/61677466/pytorch-batch-indexing
        # https://towardsdatascience.com/lstm-text-classification-using-pytorch-2c6c657f8fc0
        # To select only one element per batch we need to enumerate batch indices
        # Note: X_lengths cannot be of float or int type.
        # either torch.arange or vanilla range works for batch indexing here.
        # This also works: lstm_out_forward = lstm_out[range(batch_size), X_lengths -1, : ]
        # shape of lstm_out_forward: batch_size, hidden_dim
        lstm_out_forward = drop_out[torch.arange(batch_size), X_lengths - 1]

        tag_scores = self.hidden2tag(lstm_out_forward)
        tag_scores_flat = torch.squeeze(tag_scores, 1)

        return tag_scores_flat
