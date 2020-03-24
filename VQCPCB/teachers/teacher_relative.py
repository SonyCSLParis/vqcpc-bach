from torch import nn
from torch.nn import ModuleList, LayerNorm
from torch.nn.modules import Transformer, TransformerEncoder, TransformerEncoderLayer
import torch

from VQCPCB.transformer_custom import TransformerEncoderLayerCustom, TransformerEncoderCustom
from VQCPCB.utils import flatten


class TeacherRelative(torch.nn.Module):
    def __init__(self,
                 data_processor,
                 num_layers,
                 num_tokens_per_channel,
                 positional_embedding_size,
                 d_model,
                 dim_feedforward,
                 n_head,
                 num_tokens,
                 dropout,
                 ):
        super(TeacherRelative, self).__init__()
        self.num_channels = len(num_tokens_per_channel)
        self.data_processor = data_processor
        input_dim = data_processor.embedding_size
        assert num_tokens % self.num_channels == 0
        self.channel_embeddings = nn.Parameter(
            torch.randn((1,
                         self.num_channels,
                         positional_embedding_size))
        )

        self.num_layers = num_layers

        self.linear_to_input_transformer = nn.Linear(
            input_dim,
            d_model - positional_embedding_size
        )
        encoder_layer = TransformerEncoderLayerCustom(
            d_model=d_model,
            nhead=n_head,
            attention_bias_type='relative_attention',
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_events=num_tokens // self.num_channels,
            num_channels=self.num_channels
        )
        self.transformer = TransformerEncoderCustom(
            encoder_layer=encoder_layer,
            num_layers=self.num_layers,
        )
        self.num_tokens_per_channel = num_tokens_per_channel

        self.pre_softmaxes = nn.ModuleList([nn.Linear(d_model, num_notes)
                                            for num_notes in num_tokens_per_channel
                                            ]
                                           )

    def forward(self, x):
        """

        :param x: (batch_size, num_events, num_channels, input_dim)
        :return: list of num_channels logits (batch_size, num_events, num_tokens_of_channel)
        """
        x = self.linear_to_input_transformer(x)
        embedded_seq = flatten(x)

        batch_size = embedded_seq.size(0)
        num_tokens = embedded_seq.size(1)
        num_events = num_tokens // self.num_channels

        # channel embeddings
        embedded_seq = torch.cat(
            [embedded_seq, self.channel_embeddings.repeat(batch_size, num_events, 1)],
            dim=2
        )

        embedded_seq = embedded_seq.transpose(0, 1)
        output, _ = self.transformer(embedded_seq)
        output = output.transpose(0, 1).contiguous()
        output = output.view(batch_size,
                             num_events,
                             self.num_channels,
                             -1)
        weights_per_category = [
            pre_softmax(t[:, :, 0, :])
            for t, pre_softmax in zip(output.split(1, 2), self.pre_softmaxes)
        ]
        return weights_per_category
