import numpy as np
import torch
from torch import nn
from torch.nn.modules import TransformerEncoderLayer, TransformerEncoder


class AuxiliaryDecoder(nn.Module):
    """
    Bidirectional Transformer with upscaling
    """
    def __init__(self,
                 num_tokens_per_channel,
                 codebook_dim,
                 upscale_factors,
                 list_of_num_layers,
                 n_head,
                 d_model,
                 dim_feedforward,
                 num_tokens_bottleneck,
                 dropout):
        super(AuxiliaryDecoder, self).__init__()
        assert len(list_of_num_layers) == len(upscale_factors)
        self.num_notes_per_voice = num_tokens_per_channel
        self.num_tokens_per_block = len(self.num_notes_per_voice)
        self.d_model = d_model
        self.codebook_dim = codebook_dim
        self.upscale_factors = upscale_factors

        # self.code_embedding = nn.Embedding(self.codebook_dim, self.d_model)
        self.linear = nn.Linear(self.codebook_dim, self.d_model)

        # TODO factorised positional embeddings
        positional_embedding_size = self.d_model

        self.positional_embeddings = nn.Parameter(
            torch.randn((1,
                         num_tokens_bottleneck,
                         positional_embedding_size))
        )

        self.upscale_embeddings = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(upscale, self.d_model)
                )
                for upscale in self.upscale_factors
            ]
        )

        # self.code_embedding_dim = self.d_model - positional_embedding_size
        # TODO for now sum positional embedding
        self.code_embedding_dim = self.d_model - positional_embedding_size

        encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        # NOTE layer_norm is already contained in encoder_layers
        self.transformers = nn.ModuleList(
            [TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_layers,
            )
                for num_layers in list_of_num_layers
            ]
        )

        self.pre_softmaxes = nn.ModuleList([nn.Linear(self.d_model, num_notes)
                                            for num_notes in num_tokens_per_channel
                                            ]
                                           )

    def forward(self, input):
        """
        :param input: sequence of codebooks (batch_size, num_tokens_bottleneck, codebook_dim)
        :return: list of weights_per_channel [(b, t, d)]
        """
        # embedded_seq = self.code_embedding(input)
        embedded_seq = self.linear(input)

        batch_size = embedded_seq.size(0)
        num_tokens = embedded_seq.size(1)

        # positional embedding
        # TODO concat
        embedded_seq = embedded_seq + self.positional_embeddings.repeat(batch_size, 1, 1)

        output = embedded_seq.transpose(0, 1)
        for transformer, upscale_factor, upscale_embeddings \
                in zip(self.transformers, self.upscale_factors, self.upscale_embeddings):
            output = transformer(output)

            # upscale
            output = self.upscale(output,
                                  upscale_factor=upscale_factor,
                                  upscale_embeddings=upscale_embeddings)

        output = output.transpose(0, 1).contiguous()

        num_events = num_tokens * np.prod(self.upscale_factors) // self.num_tokens_per_block
        output = output.view(batch_size,
                             num_events,
                             self.num_tokens_per_block,
                             -1)
        weights_per_category = [
            pre_softmax(t[:, :, 0, :])
            for t, pre_softmax in zip(output.split(1, 2), self.pre_softmaxes)
        ]
        return weights_per_category

    @staticmethod
    def upscale(input, upscale_factor, upscale_embeddings):
        # WARNING
        # input is time first!
        assert len(upscale_embeddings) == upscale_factor

        sequence_length, batch_size, _ = input.size()

        upscale_embeddings = upscale_embeddings.unsqueeze(1).repeat(sequence_length,
                                                                    batch_size,
                                                                    1)
        input = torch.repeat_interleave(input, upscale_factor, dim=0)

        output = input + upscale_embeddings
        return output
