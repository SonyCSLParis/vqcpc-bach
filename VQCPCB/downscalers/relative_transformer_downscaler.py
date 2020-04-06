import numpy as np
import torch
from torch import nn

from VQCPCB.downscalers.downscaler import Downscaler
from VQCPCB.transformer.transformer_custom import TransformerEncoderLayerCustom, TransformerEncoderCustom


class RelativeTransformerDownscaler(Downscaler):
    """
    From (batch_size, num_tokens, embedding_dim)
      to (batch_size, num_tokens // prod(downscale_factors), codebook_dim)
    Uses positional embeddings
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 num_channels,
                 downscale_factors,
                 d_model,
                 n_head,
                 list_of_num_layers,
                 dim_feedforward,
                 dropout
                 ):
        super(RelativeTransformerDownscaler, self).__init__(downscale_factors)

        assert len(downscale_factors) == len(list_of_num_layers), \
            'number of transfo must match number of downscaling factors'

        # Sequence specs for the input of the transfo stack
        self.sequence_length = np.prod(downscale_factors)
        positional_embedding_size = 8
        self.num_channels = num_channels
        self.num_events = self.sequence_length // self.num_channels
        # TODO useless assert?!
        assert self.sequence_length % np.prod(downscale_factors) == 0

        self.input_linear = nn.Linear(
            input_dim,
            d_model - 2 * positional_embedding_size
        )

        self.target_channel_embeddings = nn.Parameter(
            torch.randn((1,                             # batch
                         1,                             # blocks
                         self.num_channels,             # tokens
                         positional_embedding_size))    # dim
        )

        self.events_positioning_embeddings = nn.Parameter(
            torch.randn((1,                             # batch
                         1,                             # blocks
                         self.num_events,               # tokens
                         positional_embedding_size))    # dim
        )

        self.output_dim = output_dim
        self.output_linear = nn.Linear(
            d_model,
            self.output_dim)

        transformers = []
        num_events = self.num_events
        num_channels = self.num_channels
        for downscale_factor, num_layers in zip(downscale_factors, list_of_num_layers):
            encoder_layer = TransformerEncoderLayerCustom(
                d_model=d_model,
                nhead=n_head,
                attention_bias_type='relative_attention',
                num_channels=num_channels,
                num_events=num_events,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            transfo = TransformerEncoderCustom(
                encoder_layer=encoder_layer,
                num_layers=num_layers,
            )
            transformers.append(transfo)

            # Next transfo parameters
            num_events = (num_events * num_channels) // downscale_factor
            if num_channels > 1:
                assert num_channels <= downscale_factor, \
                    f'First stack of downscaler transfo has to be larger than input num channels = {num_channels}'
                num_channels = 1

        # Stack of transformers, each downscaling of a factor indicated in downscale_factors
        self.transformers = nn.ModuleList(transformers)

    def forward(self, embedded_seq):
        """
        (batch_size, sequence_length, input_dim)
        :return: (batch_size, sequence_length // prod(downscale_factors), output_dim)
        """
        batch_size, seq_len, dim = embedded_seq.shape
        assert seq_len % self.sequence_length == 0
        num_blocks = seq_len // self.sequence_length
        embedded_seq = embedded_seq.view(batch_size, num_blocks, self.sequence_length, dim)

        # Embed input
        embedded_seq = self.input_linear(embedded_seq)
        # positional embedding
        embedded_seq = torch.cat([
            embedded_seq,
            self.target_channel_embeddings.repeat(batch_size, num_blocks, self.num_events, 1),
            self.events_positioning_embeddings
                .repeat_interleave(self.num_channels, dim=2)
                .repeat(batch_size, num_blocks, 1, 1)
        ], dim=3)

        # Prepare data: (b, l, d) and Time first
        transfo_input = embedded_seq.view(batch_size * num_blocks, self.sequence_length, -1).transpose(0, 1)

        ############################################################
        # Simple transfo ?
        # output, attentions = self.transformer(transfo_input)
        # OR
        # Downscaling stack
        for transfo, downscaling in zip(self.transformers, self.downscale_factors):
            output, attentions = transfo(transfo_input)
            transfo_input = output[::downscaling]
        output = transfo_input
        ############################################################

        assert output.shape[0] == 1
        # Take the last output token as embedding and reshape
        output = output[0].view(batch_size, num_blocks, -1)
        # project to output_dim
        output = self.output_linear(output)
        return output