import math
import torch
from torch import nn
import numpy as np

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        """
        :param d_model: dimension of the vector
        :param vocab_size: how many words in the vocabulary
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        self.embedding(x)*math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """
        For even PE(pos, 2i) = sin (pos/ 10000^(2i/d_model))
        For odd PE(pos, 2i+1) = cos (pos/ 10000^(2i/d_model))

        :param d_model:
        :param seq_len:
        :param dropout:
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len # max length of the sentence
        self.dropout = nn.Dropout(dropout) # avoids overfitting

        # create a matrix of shape (seq_len x d_model)
        pe = torch.zeros(self.seq_len, self.d_model)

        # create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/ d_model))

        # apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        self.register_buffer('pe', self.pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim=-1, keepdims=True)

        return (self.alpha*(x - mean)/(std + self.eps)) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model) # W2 and B1

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))












