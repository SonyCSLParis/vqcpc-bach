from torch import nn


class MlpUpscaler(nn.Module):
    """
    From (batch_size, num_tokens, embedding_dim)
      to (batch_size, num_tokens // prod(downscale_factors), codebook_dim)
    Uses positional embeddings
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_size,
                 dropout,
                 ):
        super(MlpUpscaler, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_size, bias=True),
                                 nn.Dropout(p=dropout),
                                 # nn.ReLU(),
                                 nn.SELU(),
                                 nn.Linear(hidden_size, output_dim, bias=True)
                                 )

    def forward(self, inputs):
        """

        :param inputs: (batch, num_blocks, dim)
        :return: z: (batch, num_blocks, dim)
        """
        outputs = self.mlp(inputs)
        return outputs