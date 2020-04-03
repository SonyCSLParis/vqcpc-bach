import numpy as np
import torch
from torch import nn

from VQCPCB.data_processor.data_processor import DataProcessor
from VQCPCB.dataloaders.dataloader_generator import DataloaderGenerator
from VQCPCB.decoders.decoder import Decoder
from VQCPCB.utils import cuda_variable, categorical_crossentropy, flatten


class DecoderAbsolute(Decoder):
    def __init__(self,
                 model_dir,
                 dataloader_generator: DataloaderGenerator,
                 data_processor: DataProcessor,
                 encoder,
                 d_model,
                 num_encoder_layers,
                 num_decoder_layers,
                 n_head,
                 dim_feedforward,
                 positional_embedding_size,
                 dropout):
        super(DecoderAbsolute, self).__init__(
            model_dir=model_dir,
            dataloader_generator=dataloader_generator,
            data_processor=data_processor,
            encoder=encoder,
            d_model=d_model
        )

        num_tokens_source = self.num_tokens_target // np.prod(self.encoder.downscaler.downscale_factors)

        self.source_positional_embeddings = nn.Parameter(
            torch.randn((1,
                         num_tokens_source,
                         positional_embedding_size))
        )

        self.target_positional_embeddings = nn.Parameter(
            torch.randn((1,
                         self.num_tokens_target,
                         positional_embedding_size))
        )

        self.embedding_dim = self.d_model - positional_embedding_size

        codebook_size = self.encoder.quantizer.codebook_size ** \
                        self.encoder.quantizer.num_codebooks
        self.source_embeddings = nn.Embedding(
            codebook_size, self.embedding_dim
        )

        #  Transformer
        self.transformer = TransformerCustom(
            num_decoder_layers=num_decoder_layers,
            num_encoder_layers=num_encoder_layers,
            d_model=self.d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Target embeddings is in data_processor
        self.linear_target = nn.Linear(self.data_processor.embedding_size
                                       + positional_embedding_size,
                                       self.d_model)

        self.sos = nn.Parameter(torch.randn((1, 1, self.d_model)))

        self.pre_softmaxes = nn.ModuleList([nn.Linear(self.d_model, num_tokens_of_channel)
                                            for num_tokens_of_channel in self.num_tokens_per_channel
                                            ]
                                           )
        # optim
        self.optimizer = None

    def __repr__(self):
        return 'DecoderAbsolute'

    def forward(self, source, target):
        """
        :param source: sequence of codebooks (batch_size, s_s)
        :param target: sequence of tokens (batch_size, num_events, num_channels)
        :return:
        """
        batch_size = source.size(0)
        # embed source
        source_seq = self.source_embeddings(source)

        # embed target
        target = self.data_processor.preprocess(target)
        target_embedded = self.data_processor.embed(target)
        target_seq = flatten(target_embedded)

        # add positional embeddings
        source_seq = torch.cat([
            source_seq,
            self.source_positional_embeddings.repeat(batch_size, 1, 1)
        ], dim=2)
        target_seq = torch.cat([
            target_seq,
            self.target_positional_embeddings.repeat(batch_size, 1, 1)
        ], dim=2)
        target_seq = self.linear_target(target_seq)

        source_seq = source_seq.transpose(0, 1)
        target_seq = target_seq.transpose(0, 1)

        # shift target_seq by one
        dummy_input = self.sos.repeat(1, batch_size, 1)
        target_seq = torch.cat(
            [
                dummy_input,
                target_seq[:-1]
            ],
            dim=0)

        # Implicitely, we do a Full//Full//Causal attention
        target_mask = cuda_variable(
            self._generate_square_subsequent_mask(target_seq.size(0))
        )

        output = self.transformer(source_seq,
                                  target_seq,
                                  tgt_mask=target_mask
                                  )

        output = output.transpose(0, 1).contiguous()

        output = output.view(batch_size,
                             -1,
                             self.num_channels,
                             self.d_model)
        weights_per_category = [
            pre_softmax(t[:, :, 0, :])
            for t, pre_softmax in zip(output.split(1, 2), self.pre_softmaxes)
        ]

        # we can change loss mask
        loss = categorical_crossentropy(
            value=weights_per_category,
            target=target,
            mask=torch.ones_like(target)
        )

        loss = loss.mean()
        return {
            'loss': loss,
            'attentions_decoder': attentions_decoder,
            'attentions_encoder': attentions_encoder,
            'weights_per_category': weights_per_category,
            'monitored_quantities': {
                'loss': loss.item()
            }
        }