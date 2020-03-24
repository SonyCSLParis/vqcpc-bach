import torch
from torch import nn
from torch.nn.modules import TransformerEncoderLayer

from VQCPCB.downscalers.downscaler import Downscaler
from VQCPCB.utils import flatten


class TransformerDownscaler(Downscaler):
    """
    From (batch_size, num_tokens, embedding_dim)
      to (batch_size, num_tokens // prod(downscale_factors), codebook_dim)
    Uses positional embeddings
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 downscale_factors,
                 num_tokens,
                 d_model,
                 n_head,
                 list_of_num_layers,
                 dim_feedforward,
                 attention_masking_type,
                 dropout
                 ):
        super(TransformerDownscaler, self).__init__(downscale_factors)
        assert len(downscale_factors) == len(list_of_num_layers)

        positional_embedding_size = 8
        self.sequence_length = num_tokens
        self.positional_embeddings = nn.Parameter(
            torch.randn((1,
                         num_tokens,
                         positional_embedding_size))
        )

        self.output_dim = output_dim

        self.input_linear = nn.Linear(
            input_dim,
            d_model - positional_embedding_size
        )

        self.output_linear = nn.Linear(
            d_model,
            self.output_dim)

        # TODO same TransformerEncoderLayer?
        # TODO relative attention
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformers = nn.ModuleList([
            TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_layers,
            )
            for num_layers in list_of_num_layers]
        )

        self.attention_masking_type = attention_masking_type

    def forward(self, embedded_seq):
        """
        (batch_size, sequence_length, input_dim)
        :return: (batch_size, sequence_length // prod(downscale_factors), output_dim)
        """
        batch_size = embedded_seq.size(0)

        # to d_model - positional_embedding_size
        embedded_seq = self.input_linear(embedded_seq)
        # positional embedding
        embedded_seq = torch.cat(
            [embedded_seq, self.positional_embeddings.repeat(batch_size, 1, 1)],
            dim=2
        )

        output = embedded_seq.transpose(0, 1)
        attention_masks = self.get_attention_masks()
        for transformer, downscale, mask in zip(self.transformers,
                                                self.downscale_factors,
                                                attention_masks):
            output = transformer(output, mask=mask)
            # downscale
            output = output[::downscale]

        output = output.transpose(0, 1).contiguous()

        # project to output_dim
        output = self.output_linear(output)
        return output

    def get_attention_masks(self):
        """
        Returns list of src_attn_mask for each block of transformers
        Depends on attention_masking_type
        :return:
        """
        if self.attention_masking_type is None:
            return [None] * len(self.downscale_factors)
        elif self.attention_masking_type == 'block':
            block_sizes = [
                int(np.prod(self.downscale_factors[i:]))
                for i in range(len(self.downscale_factors))
            ]
            sequence_sizes = [
                self.sequence_length // int(np.prod(self.downscale_factors[:i]))
                for i in range(len(self.downscale_factors))
            ]
            return [
                self._block_attention_mask(block_size=block_size,
                                           sequence_size=sequence_size)
                for block_size, sequence_size in zip(block_sizes, sequence_sizes)
            ]

        else:
            return NotImplementedError

    @staticmethod
    def _block_attention_mask(block_size, sequence_size):
        assert sequence_size % block_size == 0
        mask = torch.eye(sequence_size // block_size) \
            .repeat_interleave(block_size, dim=0). \
            repeat_interleave(block_size, dim=1)
        return mask.to('cuda')
