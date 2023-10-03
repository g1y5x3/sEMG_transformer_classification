# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# https://github.com/gzerveas/mvts_transformer/blob/3f2e378bc77d02e82a44671f20cf15bc7761671a/src/models/ts_transformer.py#L105-L125
class LearnablePositionalEncoding(nn.Module):
  def __init__(self, d_model: int, dropout: float =0.1, max_len: int = 4000):
    super(LearnablePositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)
    self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))
    nn.init.uniform_(self.pe, -0.02, 0.02)

  def forward(self, x):
    """
    Arguments:
      x: Tensor, shape [seq_len, batch_size, embedding_dim]
    Outputs:
      output: Tensor, shape [seq_len, batch_size, embedding_dim]
    """
    x = x + self.pe[:x.size(0), :]
    return self.dropout(x)

class TransformerModel(nn.Module):
  # In normal transformer's context, 'feat_dim' would be 'n_token'
  def __init__(self, feat_dim: int, d_model: int, n_head: int, d_hid: int, n_layer: int, 
               dropout: float = 0.1, max_len: int = 4000, n_class: int = 2):
    super().__init__()
    self.d_model = torch.tensor(d_model)
    # since each input always has the same seq_len, we don't need the look-up table from nn.Embedding 
    self.project = nn.Linear(feat_dim, d_model)
    self.pos_encoder = LearnablePositionalEncoding(d_model, dropout, max_len)
    encoder_layers = TransformerEncoderLayer(d_model, n_head, d_hid, dropout)
    self.transformer_encoder = TransformerEncoder(encoder_layers, n_layer)
    self.activation = F.gelu
    self.dropout = nn.Dropout(dropout)
    self.output_layer = nn.Linear(d_model*max_len, n_class)
     
  def forward(self, x: Tensor, x_mask: Tensor = None) -> Tensor:
    """
    Arguments:
      x: Tensor, shape [batch_size, seq_len, feat_dim]
    Outputs:
      output: Tensor, shape [batch_size, n_class]
    """
    # permute because pytorch convention for transformers is [seq_len, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
    x = x.permute(1,0,2)
    # embedding without the need for look-up table
    x = self.project(x) * torch.sqrt(self.d_model)
    x = self.pos_encoder(x)
    # TODO: when input samples have considerable variation in length, appropriate is needed
    output = self.transformer_encoder(x)
    output = self.activation(output)
    output = output.permute(1,0,2)
    output = self.dropout(output)
    # TODO: add masking for input with variate length
    output = output.reshape(output.shape[0],-1)
    output = self.output_layer(output)

    return output