import numpy as np
import torch
from torch import nn

from VQCPCB.downscalers.downscaler import Downscaler


class LstmDownscaler(Downscaler):
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
                 hidden_size,
                 num_layers,
                 dropout,
                 bidirectional
                 ):
        super(LstmDownscaler, self).__init__(downscale_factors)
        assert len(downscale_factors) == 1
        self.output_dim = output_dim
        self.num_channels = num_channels

        self.g_enc_fwd = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        if bidirectional:
            self.g_enc_bwd = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=True,
                batch_first=True,
                dropout=dropout,
                bidirectional=False,
            )
            output_linear_input_dim = hidden_size * 2
        else:
            self.g_enc_bwd = None
            output_linear_input_dim = hidden_size

        self.output_linear = nn.Linear(output_linear_input_dim, output_dim, bias=True)

    def forward(self, inputs):
        """

        :param inputs: (batch, seq_len, dim)
        with seq = num_blocks * block_size
        :return: z: (batch,
        """
        #  split in cpc blocks
        batch_size, seq_len, dim = inputs.shape
        num_blocks = seq_len // self.downscale_factors[0]
        assert seq_len % self.downscale_factors[0] == 0
        inputs = inputs.view(batch_size, num_blocks, self.downscale_factors[0], dim)

        z = self.compute_z(inputs)
        return z

    def compute_z(self, x):
        seq_len, input_size = x.size()[-2:]
        batch_dims = x.size()[:-2]
        product_batch_dims = np.prod(batch_dims)

        x = x.view(product_batch_dims, seq_len, input_size)
        z_seq_fwd, _ = self.g_enc_fwd(x)
        z_last_fwd = z_seq_fwd[:, -1]

        #  backward if bidirectional
        if self.g_enc_bwd is not None:
            z_seq_bwd, _ = self.g_enc_bwd(x.flip(dims=[1]))
            z_last_bwd = z_seq_bwd[:, -1]
            z_bi = torch.cat([z_last_fwd, z_last_bwd], dim=-1)
        else:
            z_bi = z_last_fwd

        # linear
        z = self.output_linear(z_bi)

        z = z.view(*batch_dims, -1)
        return z
