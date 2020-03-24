import torch

from torch import nn
import numpy as np

from VQCPCB.transformer_custom import TransformerEncoderLayerCustom, TransformerEncoderCustom


class RelativeTransformerDownscaler(nn.Module):
    """
    From (batch_size, num_tokens, embedding_dim)
      to (batch_size, num_tokens // prod(downscale_factors), codebook_dim)
    Uses channel/event_in_code embeddings
    That's why we need to pass a num_channels argument ;)
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 downscale_factors,
                 num_tokens,
                 num_channels,
                 d_model,
                 n_head,
                 list_of_num_layers,
                 dim_feedforward,
                 attention_masking_type,
                 attention_bias_type,
                 dropout
                 ):
        super(RelativeTransformerDownscaler, self).__init__()
        self.downscale_factors = downscale_factors
        self.num_channels = num_channels
        self.num_tokens = num_tokens
        self.total_downscaling = int(np.prod(self.downscale_factors))
        self.num_events_per_code = self.total_downscaling // self.num_channels

        assert len(downscale_factors) == len(list_of_num_layers)
        assert self.total_downscaling % self.num_channels == 0

        positional_embedding_size = 8
        self.channel_embeddings = nn.Parameter(
            torch.randn((1,
                         self.num_channels,
                         positional_embedding_size))
        )
        self.events_positioning_embeddings = nn.Parameter(
            torch.randn(1,
                        self.num_events_per_code,
                        positional_embedding_size)
        )

        self.output_dim = output_dim

        self.input_linear = nn.Linear(
            input_dim,
            d_model - 2 * positional_embedding_size
        )

        self.output_linear = nn.Linear(
            d_model,
            self.output_dim)

        # create encoding layers
        transformer_list = []
        num_channels_for_encoder = self.num_channels
        num_events_for_encoder = self.num_tokens // self.num_channels
        for num_layers, downscale_factor in zip(list_of_num_layers, downscale_factors):
            encoder_layer = TransformerEncoderLayerCustom(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=dim_feedforward,
                attention_bias_type=attention_bias_type,
                dropout=dropout,
                num_channels=num_channels_for_encoder,
                num_events=num_events_for_encoder
            )
            transformer = TransformerEncoderCustom(
                encoder_layer=encoder_layer,
                num_layers=num_layers,
            )
            transformer_list.append(transformer)

            if num_channels_for_encoder > 1:
                assert num_channels_for_encoder % downscale_factor == 0
                num_channels_for_encoder = num_channels_for_encoder // downscale_factor
            else:
                assert num_events_for_encoder % downscale_factor == 0
                num_events_for_encoder = num_events_for_encoder // downscale_factor

        self.transformers = nn.ModuleList(transformer_list)

        self.attention_masking_type = attention_masking_type

    def forward(self, embedded_seq):
        """
        (batch_size, sequence_length, input_dim)
        :return: (batch_size, sequence_length // prod(downscale_factors), output_dim)
        """
        batch_size = embedded_seq.size(0)
        num_tokens_input = embedded_seq.size(1)
        assert num_tokens_input % self.total_downscaling == 0
        assert num_tokens_input % self.num_channels == 0

        # to d_model - positional_embedding_size
        embedded_seq = self.input_linear(embedded_seq)
        # positional embedding
        embedded_seq = torch.cat(
            [embedded_seq,
             self.channel_embeddings.repeat(batch_size, num_tokens_input // self.num_channels, 1),
             self.events_positioning_embeddings
                 .repeat_interleave(self.num_channels, dim=1)
                 .repeat(batch_size, num_tokens_input // self.total_downscaling, 1)
             ],
            dim=2
        )

        output = embedded_seq.transpose(0, 1)
        attention_masks = self.get_attention_masks(num_tokens=num_tokens_input)
        for transformer, downscale, mask in zip(self.transformers,
                                                self.downscale_factors,
                                                attention_masks):
            output, _ = transformer(output, mask=mask)
            # downscale
            output = output[::downscale]

        output = output.transpose(0, 1).contiguous()

        # project to output_dim
        output = self.output_linear(output)
        return output

    def get_attention_masks(self, num_tokens):
        """
        Returns list of src_attn_mask for each block of transformers
        Depends on attention_masking_type
        :param num_tokens: num_tokens of the input sequence
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
                num_tokens // int(np.prod(self.downscale_factors[:i]))
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
