import os
from datetime import datetime
from itertools import islice

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from VQCPCB.dataloaders.dataloader_generator import DataloaderGenerator
from VQCPCB.transformer.transformer_custom import TransformerEncoderCustom, \
    TransformerEncoderLayerCustom
from VQCPCB.utils import dict_pretty_print, cuda_variable, categorical_crossentropy, to_numpy


class PriorRelative(nn.Module):
    def __init__(self,
                 model_dir,
                 dataloader_generator: DataloaderGenerator,
                 encoder,
                 d_model,
                 num_layers,
                 n_head,
                 dim_feedforward,
                 embedding_size,
                 num_channels,
                 num_events,
                 dropout):
        """
        Like DecoderRelative, but without conditioning
        uses of course data_processor form encoder
        :param model_dir:
        :param dataloader_generator:
        :param encoder:
        :param d_model:
        :param num_layers:
        :param n_head:
        :param dim_feedforward:
        :param embedding_size:
        :param num_channels:
        :param num_events:
        :param dropout:
        """

        super(PriorRelative, self).__init__()
        self.model_dir = model_dir
        self.encoder = encoder
        # freeze encoder
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.dataloader_generator = dataloader_generator

        self.num_tokens_per_channel = [
            encoder.quantizer.codebook_size ** encoder.quantizer.num_codebooks]
        self.num_channels = num_channels
        # for now we only use 1 channel for quantized codes
        assert self.num_channels == 1
        # no positional embedding in this case
        # TODO factorised positional embeddings
        self.d_model = d_model
        self.num_tokens = num_channels * num_events

        #  Transformer
        encoder_layer = TransformerEncoderLayerCustom(
            d_model=d_model,
            nhead=n_head,
            attention_bias_type='relative_attention',
            num_channels=num_channels,
            num_events=num_events,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        self.transformer = TransformerEncoderCustom(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        # Target embeddings is in data_processor
        self.embedding = nn.Embedding(
            self.num_tokens_per_channel[0],
            embedding_size
        )
        self.linear = nn.Linear(embedding_size,
                                self.d_model)

        self.sos = nn.Parameter(torch.randn((1, 1, self.d_model)))

        self.pre_softmaxes = nn.ModuleList(
            [nn.Linear(self.d_model, num_tokens_of_channel)
             for num_tokens_of_channel in self.num_tokens_per_channel]
        )
        # optim
        self.optimizer = None

    def init_optimizers(self, lr=1e-3):
        # TODO use radam
        self.optimizer = torch.optim.Adam(
            list(self.parameters()),
            lr=lr
        )

    def __repr__(self):
        return 'PriorRelative'

    def save(self):
        # This saves also the encoder
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        torch.save(self.state_dict(), f'{self.model_dir}/prior')
        self.encoder.save()
        # print(f'Model {self.__repr__()} saved')

    def load(self, device):
        print(f'Loading models {self.__repr__()}')
        self.load_state_dict(torch.load(f'{self.model_dir}/prior', map_location=torch.device(device)))
        self.encoder.load()

    def forward(self, x):
        """
        :param x: sequence of codebooks (batch_size, s_s)
        :return:
        """
        batch_size = x.size(0)

        target = x.unsqueeze(dim=2)
        # embed
        x_seq = self.embedding(x)
        x_seq = self.linear(x_seq)

        # add positional embeddings
        x_seq = x_seq.transpose(0, 1)

        # shift target_seq by one
        dummy_input = self.sos.repeat(1, batch_size, 1)
        x_seq = torch.cat(
            [
                dummy_input,
                x_seq[:-1]
            ],
            dim=0)

        mask = cuda_variable(
            self._generate_square_subsequent_mask(x_seq.size(0))
        )

        # for custom
        output, attentions = self.transformer(x_seq,
                                              mask=mask
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
            'loss':                 loss,
            'weights_per_category': weights_per_category,
            'monitored_quantities': {
                'loss': loss.item()
            }
        }

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def epoch(self, data_loader,
              train=True,
              num_batches=None,
              ):
        means = None

        if train:
            self.train()
            self.encoder.eval()
        else:
            self.eval()

        for sample_id, tensor_dict in tqdm(enumerate(
                islice(data_loader, num_batches)),
                ncols=80):

            # ======== ==================
            with torch.no_grad():
                x = tensor_dict['x']
                # compute encoding_indices version
                z_quantized, encoding_indices, quantization_loss = self.encoder(x)

            # ========Train decoder =============
            self.optimizer.zero_grad()
            forward_pass = self.forward(
                encoding_indices
            )
            loss = forward_pass['loss']
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 5)
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 5)
                self.optimizer.step()

            # Monitored quantities
            monitored_quantities = forward_pass['monitored_quantities']

            # average quantities
            if means is None:
                means = {key: 0
                         for key in monitored_quantities}
            means = {
                key: value + means[key]
                for key, value in monitored_quantities.items()
            }

            del loss

        # renormalize monitored quantities
        means = {
            key: value / num_batches
            for key, value in means.items()
        }
        return means

    def train_model(self,
                    batch_size,
                    num_batches=None,
                    num_epochs=10,
                    lr=1e-3,
                    plot=False,
                    num_workers=0,
                    **kwargs
                    ):
        # (generator_train,
        #  generator_val,
        #  generator_test) = self.dataset.data_loaders(batch_size=batch_size)
        if plot:
            self.writer = SummaryWriter(f'{self.model_dir}')

        best_val = 1e8
        self.init_optimizers(lr=lr)
        for epoch_id in range(num_epochs):
            (generator_train,
             generator_val,
             generator_test) = self.dataloader_generator.dataloaders(
                batch_size=batch_size,
                num_workers=num_workers)

            monitored_quantities_train = self.epoch(
                data_loader=generator_train,
                train=True,
                num_batches=num_batches,
            )
            del generator_train

            monitored_quantities_val = self.epoch(
                data_loader=generator_val,
                train=False,
                num_batches=num_batches // 2 if num_batches is not None else None,
            )
            del generator_val

            valid_loss = monitored_quantities_val['loss']
            # self.scheduler.step(monitored_quantities_val["loss"])

            print(f'======= Epoch {epoch_id} =======')
            print(f'---Train---')
            dict_pretty_print(monitored_quantities_train, endstr=' ' * 5)
            print()
            print(f'---Val---')
            dict_pretty_print(monitored_quantities_val, endstr=' ' * 5)
            print('\n')

            if valid_loss < best_val:
                self.save()
                best_val = valid_loss

            if plot:
                self.plot(epoch_id,
                          monitored_quantities_train,
                          monitored_quantities_val)

    def plot(self, epoch_id, monitored_quantities_train,
             monitored_quantities_val):
        for k, v in monitored_quantities_train.items():
            self.writer.add_scalar(f'{k}/train', v, epoch_id)
        for k, v in monitored_quantities_val.items():
            self.writer.add_scalar(f'{k}/val', v, epoch_id)

    def generate(self, num_tokens, decoder, temperature=1.,
                 num_generated_codes=1,
                 num_decodings_per_generating_code=1,
                ):
        self.eval()
        decoder.eval()
        with torch.no_grad():
            # init
            x = cuda_variable(
                torch.zeros(1, num_tokens, self.num_channels)
            ).long()

            x = x.repeat(num_generated_codes, 1, 1)
            assert num_tokens % self.num_channels == 0
            # num_tokens is the number of the sequence to be generated
            # while self.num_tokens is the number of tokens of the input of the model
            assert num_tokens >= self.num_tokens
            num_events = num_tokens // self.num_channels

            for event_index in range(num_events):
                for channel_index in range(self.num_channels):
                    # removes channel dim
                    x_input = x[:, :, 0]
                    if event_index >= self.num_tokens:
                        x_input = x_input[:, event_index - self.num_tokens + 1: event_index + 1]
                        event_offset = event_index - self.num_tokens + 1
                    else:
                        x_input = x_input[:, :self.num_tokens]
                        event_offset = 0

                    weights_per_voice = self.forward(x_input)['weights_per_category']

                    weights = weights_per_voice[channel_index]
                    probs = torch.softmax(
                        weights[:, event_index - event_offset, :],
                        dim=1)
                    p = to_numpy(probs)
                    # temperature ?!
                    p = np.exp(np.log(p + 1e-20) * temperature)
                    p = p / p.sum(axis=1, keepdims=True)

                    for batch_index in range(num_generated_codes):
                        new_pitch_index = np.random.choice(np.arange(
                            self.num_tokens_per_channel[channel_index]
                        ), p=p[batch_index])
                        x[batch_index, event_index, channel_index] = int(new_pitch_index)

        source = x[:, :, 0]
        scores = decoder.generate_from_code_long(encoding_indices=source,
                                                 temperature=temperature,
                                                 num_decodings=num_decodings_per_generating_code)

        # save scores in model_dir
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if not os.path.exists(f'{self.model_dir}/generations'):
            os.mkdir(f'{self.model_dir}/generations')

        for k, score in enumerate(scores):
            score.write('xml', f'{self.model_dir}/generations/{timestamp}_{k}.xml')

        return scores