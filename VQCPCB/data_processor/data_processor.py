import torch
from torch import nn

from VQCPCB.utils import cuda_variable


class DataProcessor(nn.Module):
    """
    Abstract class used for preprocessing and embedding
    Preprocessing: from ? -> (batch_size, num_events, num_channels)
    Embedding: from (batch_size, num_events, num_channels) ->
      (batch_size, num_events, num_channels, embedding_size)
    """

    def __init__(self, embedding_size,
                 num_events,
                 num_tokens_per_channel,
                 add_mask_token=True):
        super(DataProcessor, self).__init__()
        self.embedding_size = embedding_size
        self.num_events = num_events
        self.num_tokens_per_channel = num_tokens_per_channel
        self.num_tokens = self.num_events * len(self.num_tokens_per_channel)
        self.num_channels = len(self.num_tokens_per_channel)

        additional_token = 1 if add_mask_token else 0
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings + additional_token, self.embedding_size)
                for num_embeddings in self.num_tokens_per_channel
            ]
        )

    def embed(self, x):
        """

        :param x: (..., num_channels)
        :return: (..., num_channels, embedding_size)
        """
        return torch.cat([
            embedding(t)
            for t, embedding in
            zip(x.split(1, dim=-1), self.embeddings)
        ], dim=-2
        )

    def embed_dict(self, tensor_dict):
        """
        to be called after preprocess

        :param tensor_dict: dict of tensors of shape (... num_events, num_channels)
        :return:
        """
        return {
            k: self.embed(v)
            for k, v in tensor_dict.items()
        }

    def preprocess_dict(self, tensor_dict):
        """
        put to cuda and format as (... num_events, num_channels)
        :param tensor_dict:
        :return:
        """
        return {
            k: self.preprocess(v)
            for k, v in tensor_dict.items()
        }

    def preprocess(self, x):
        """
        Subclasses can only reimplement this method
        This is not necessary

        :param x: ? -> (batch_size, num_events, num_channels)
        :return:
        """
        return cuda_variable(x.long())

    def postprocess(self, original, reconstruction):
        """
        Inverse of preprocess

        :param x: (batch_size, num_events, num_channels) -> ?
        :return:
        """
        return reconstruction

    def dump(self, x):
        """

        :param x: (num_events, num_channels)
        :return:
        """
        x = x.contiguous().view(-1).detach().cpu().numpy()
        return '_'.join([str(c).zfill(2) for c in x])
