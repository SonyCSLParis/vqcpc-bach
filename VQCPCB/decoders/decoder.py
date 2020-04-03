import os
from datetime import datetime
from itertools import islice

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from VQCPCB.data_processor.data_processor import DataProcessor
from VQCPCB.dataloaders.dataloader_generator import DataloaderGenerator
from VQCPCB.datasets.helpers import PAD_SYMBOL, START_SYMBOL, END_SYMBOL
from VQCPCB.transformer.transformer_custom import TransformerCustom, TransformerDecoderCustom, \
    TransformerEncoderCustom, \
    TransformerDecoderLayerCustom, TransformerEncoderLayerCustom, TransformerAlignedDecoderLayerCustom
from VQCPCB.utils import cuda_variable, categorical_crossentropy, flatten, dict_pretty_print, top_k_top_p_filtering, \
    to_numpy


class Decoder(nn.Module):
    def __init__(self,
                 model_dir,
                 dataloader_generator: DataloaderGenerator,
                 data_processor: DataProcessor,
                 encoder,
                 transformer_type,  # absolute or relative
                 encoder_attention_type,  # anticausal, causal, diagonal or full
                 cross_attention_type,  # anticausal, causal, diagonal or full
                 d_model,
                 num_encoder_layers,
                 num_decoder_layers,
                 n_head,
                 dim_feedforward,
                 positional_embedding_size,
                 num_channels_encoder,
                 num_events_encoder,
                 num_channels_decoder,
                 num_events_decoder,
                 dropout):
        """
        Like DecoderCustom, but the positioning is relative
        :param model_dir:
        :param dataloader_generator:
        :param data_processor:
        :param encoder:
        :param d_model:
        :param num_encoder_layers:
        :param num_decoder_layers:
        :param n_head:
        :param dim_feedforward:
        :param positional_embedding_size:
        :param num_channels_encoder:
        :param num_events_encoder:
        :param num_channels_decoder:
        :param num_events_decoder:
        :param dropout:
        """
        super(Decoder, self).__init__()
        self.transformer_type = transformer_type
        assert encoder_attention_type in ['anticausal', 'causal', 'full']
        self.encoder_attention_type = encoder_attention_type
        assert cross_attention_type in ['anticausal', 'causal', 'diagonal', 'full']
        self.cross_attention_type = cross_attention_type
        self.model_dir = model_dir
        self.encoder = encoder
        # freeze encoder
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.dataloader_generator = dataloader_generator
        self.data_processor = data_processor

        # Compute num_tokens for source and target
        self.num_tokens_per_channel = self.data_processor.num_tokens_per_channel
        self.num_channels = len(self.num_tokens_per_channel)
        self.d_model = d_model
        self.num_tokens_target = self.data_processor.num_tokens
        self.total_upscaling = int(np.prod(self.encoder.downscaler.downscale_factors))
        assert self.num_tokens_target % self.total_upscaling == 0
        assert self.num_tokens_target == num_channels_decoder * num_events_decoder

        ######################################################
        # Embeddings
        if transformer_type == 'absolute':
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
        elif transformer_type == 'relative':
            self.target_channel_embeddings = nn.Parameter(
                torch.randn((1,
                             self.num_channels,
                             positional_embedding_size))
            )
            self.num_events_per_code = self.total_upscaling // self.num_channels
            # Position relative to a code
            self.target_events_positioning_embeddings = nn.Parameter(
                torch.randn((1,
                             self.num_events_per_code,
                             positional_embedding_size))
            )

        ######################################################
        #  Encoder Transformer
        if transformer_type == 'absolute':
            encoder_layer = TransformerEncoderLayerCustom(
                d_model=d_model,
                nhead=n_head,
                attention_bias_type=None,
                num_channels=num_channels_encoder,
                num_events=num_events_encoder,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
        elif transformer_type == 'relative':
            encoder_layer = TransformerEncoderLayerCustom(
                d_model=d_model,
                nhead=n_head,
                attention_bias_type='relative_attention',
                num_channels=num_channels_encoder,
                num_events=num_events_encoder,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )

        custom_encoder = TransformerEncoderCustom(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )

        ######################################################
        #  Decoder Transformer
        if transformer_type == 'absolute':
            decoder_layer = TransformerDecoderLayerCustom(
                d_model=d_model,
                nhead=n_head,
                attention_bias_type_self=None,
                attention_bias_type_cross=None,
                num_channels_encoder=num_channels_encoder,
                num_events_encoder=num_events_encoder,
                num_channels_decoder=num_channels_decoder,
                num_events_decoder=num_events_decoder,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
        elif transformer_type == 'relative':
            if cross_attention_type == 'diagonal':
                decoder_layer = TransformerAlignedDecoderLayerCustom(
                    d_model=d_model,
                    nhead=n_head,
                    attention_bias_type_self='relative_attention',
                    attention_bias_type_cross=None,
                    num_channels_encoder=num_channels_encoder,
                    num_events_encoder=num_events_encoder,
                    num_channels_decoder=num_channels_decoder,
                    num_events_decoder=num_events_decoder,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout
                )
            else:
                decoder_layer = TransformerDecoderLayerCustom(
                    d_model=d_model,
                    nhead=n_head,
                    attention_bias_type_self='relative_attention',
                    attention_bias_type_cross='relative_attention_target_source',
                    num_channels_encoder=num_channels_encoder,
                    num_events_encoder=num_events_encoder,
                    num_channels_decoder=num_channels_decoder,
                    num_events_decoder=num_events_decoder,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout
                )

        custom_decoder = TransformerDecoderCustom(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers
        )

        ######################################################
        # Complete Transformer
        self.transformer = TransformerCustom(
            d_model=self.d_model,
            nhead=n_head,
            custom_encoder=custom_encoder,
            custom_decoder=custom_decoder
        )

        ######################################################
        # Target pre-processing
        # (Target embeddings is in data_processor, so no need to have an emebdding here.)
        # Just a linear to map to the proper dimension
        if self.transformer_type == 'absolute':
            linear_target_input_size = self.data_processor.embedding_size + positional_embedding_size
        elif self.transformer_type == 'relative':
            linear_target_input_size = self.data_processor.embedding_size + positional_embedding_size * 2
        self.linear_target = nn.Linear(linear_target_input_size, self.d_model)
        # start of sentence
        self.sos = nn.Parameter(torch.randn((1, 1, self.d_model)))

        ######################################################
        # Source pre-processing
        # Re-embed the codes extracted by the encoder (instead of using directly the z computed by the encoder,
        # we use the cluster indices computed by the encoder and learn a new embedding jointly with the decoder)
        if self.transformer_type == 'relative':
            source_embedding_dim = self.d_model
        elif self.transformer_type == 'absolute':
            source_embedding_dim = self.d_model - positional_embedding_size
        codebook_size = self.encoder.quantizer.codebook_size ** self.encoder.quantizer.num_codebooks
        self.source_embeddings = nn.Embedding(
            codebook_size, source_embedding_dim
        )

        ######################################################
        # Output dimension adjustment
        self.pre_softmaxes = nn.ModuleList([nn.Linear(self.d_model, num_tokens_of_channel)
                                            for num_tokens_of_channel in self.num_tokens_per_channel
                                            ]
                                           )

        ######################################################
        # optim
        self.optimizer = None

    def __repr__(self):
        name_mappings = dict(
            anticausal='AC',
            causal='C',
            full='F',
            diagonal='D'
        )
        return f'Decoder-{self.transformer_type}-{name_mappings[self.encoder_attention_type]}-' \
               f'{name_mappings[self.cross_attention_type]}'

    def init_optimizers(self, lr=1e-3):
        self.optimizer = torch.optim.Adam(
            list(self.parameters())
            ,
            lr=lr
        )

    def save(self, early_stopped):
        # This saves also the encoder
        if early_stopped:
            model_dir = f'{self.model_dir}/early_stopped'
        else:
            model_dir = f'{self.model_dir}/overfitted'

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        torch.save(self.state_dict(), f'{model_dir}/decoder')
        # print(f'Model {self.__repr__()} saved')

    def load(self, early_stopped, device):
        print(f'Loading models {self.__repr__()}')
        if early_stopped:
            model_dir = f'{self.model_dir}/early_stopped'
        else:
            model_dir = f'{self.model_dir}/overfitted'
        self.load_state_dict(torch.load(f'{model_dir}/decoder', map_location=torch.device(device)))

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _generate_anticausal_mask(self, sz):
        return cuda_variable(self._generate_square_subsequent_mask(sz)).t()

    def _generate_causal_mask(self, sz):
        return cuda_variable(self._generate_square_subsequent_mask(sz))

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

            # ======== Get codes from Encoder ==================
            with torch.no_grad():
                x = tensor_dict['x']
                # compute encoding_indices version
                z_quantized, encoding_indices, quantization_loss = self.encoder(x)
                encoding_indices = self.encoder.merge_codes(encoding_indices)

            # ======== Train decoder =============
            self.optimizer.zero_grad()
            forward_pass = self.forward(
                encoding_indices,
                x
            )
            loss = forward_pass['loss']
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
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
            key: value / (sample_id + 1)
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

            self.save(early_stopped=False)
            if valid_loss < best_val:
                self.save(early_stopped=True)
                best_val = valid_loss

            if plot:
                self.plot(epoch_id,
                          monitored_quantities_train,
                          monitored_quantities_val)

    def forward(self, source, target):
        """
        :param source: sequence of codebooks (batch_size, s_s)
        :param target: sequence of tokens (batch_size, num_events, num_channels)
        :return:
        """
        batch_size = source.size(0)
        # embed
        source_seq = self.source_embeddings(source)

        target = self.data_processor.preprocess(target)
        target_embedded = self.data_processor.embed(target)
        target_seq = flatten(target_embedded)

        num_tokens_target = target_seq.size(1)

        if self.transformer_type == 'relative':
            # add positional embeddings
            target_seq = torch.cat([
                target_seq,
                self.target_channel_embeddings.repeat(batch_size,
                                                      num_tokens_target // self.num_channels,
                                                      1),
                self.target_events_positioning_embeddings
                    .repeat_interleave(self.num_channels, dim=1)
                    .repeat((batch_size, num_tokens_target // self.total_upscaling, 1))
            ], dim=2)
        elif self.transformer_type == 'absolute':
            source_seq = torch.cat([
                source_seq,
                self.source_positional_embeddings.repeat(batch_size, 1, 1)
            ], dim=2)
            target_seq = torch.cat([
                target_seq,
                self.target_positional_embeddings.repeat(batch_size, 1, 1)
            ], dim=2)

        target_seq = self.linear_target(target_seq)

        # time dim first
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

        # masks: anti-causal for encoder, causal for decoder
        source_length = source_seq.size(0)
        target_length = target_seq.size(0)
        # cross-masks
        if self.cross_attention_type in ['diagonal', 'full']:
            memory_mask = None
        elif self.cross_attention_type == 'causal':
            raise NotImplementedError
            # C'est une galere de ouf,
            # faut repeat_interleave pour faire des masques rectangulaires
            # c'est chiant....
        elif self.cross_attention_type == 'anticausal':
            raise NotImplementedError
            # pareil

        # self-encoder masks
        if self.encoder_attention_type in ['diagonal', 'full']:
            source_mask = None
        elif self.encoder_attention_type == 'causal':
            source_mask = self._generate_causal_mask(source_length)
        elif self.encoder_attention_type == 'anticausal':
            source_mask = self._generate_anticausal_mask(source_length)

        # Causal target mask
        target_mask = self._generate_causal_mask(target_length)

        # for custom
        output, attentions_decoder, attentions_encoder = self.transformer(source_seq,
                                                                          target_seq,
                                                                          tgt_mask=target_mask,
                                                                          src_mask=source_mask,
                                                                          memory_mask=memory_mask
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

    def plot(self, epoch_id, monitored_quantities_train,
             monitored_quantities_val):
        for k, v in monitored_quantities_train.items():
            self.writer.add_scalar(f'{k}/train', v, epoch_id)
        for k, v in monitored_quantities_val.items():
            self.writer.add_scalar(f'{k}/val', v, epoch_id)

    def generate(self, temperature, batch_size=1, plot_attentions=False):
        self.eval()
        (generator_train, generator_val, _) = self.dataloader_generator.dataloaders(
            batch_size=1,
            shuffle_val=True
        )

        with torch.no_grad():
            tensor_dict = next(iter(generator_val))
            # tensor_dict = next(iter(generator_train))

            x_original_single = tensor_dict['x']
            x_original = x_original_single.repeat(batch_size, 1, 1)

            # compute downscaled version
            _, encoding_indices, _ = self.encoder(x_original)
            encoding_indices = self.encoder.merge_codes(encoding_indices)

            x = self.init_generation(num_events=self.data_processor.num_events)

            # Duplicate along batch dimension
            x = x.repeat(batch_size, 1, 1)

            attentions_decoder_list = []
            attentions_encoder_list = []
            attentions_cross_list = []

            for event_index in range(self.data_processor.num_events):
                for channel_index in range(self.num_channels):
                    forward_pass = self.forward(encoding_indices,
                                                x)

                    weights_per_voice = forward_pass['weights_per_category']
                    weights = weights_per_voice[channel_index]

                    # Keep only the last token predictions of the first batch item (batch size 1), apply a
                    # temperature coefficient and filter
                    logits = weights[:, event_index, :] / temperature
                    # Remove meta symbols
                    for sym in [START_SYMBOL, END_SYMBOL, PAD_SYMBOL]:
                        sym_index = self.dataloader_generator.dataset.note2index_dicts[channel_index][sym]
                        logits[:, sym_index] = -float("inf")
                    # Top-p sampling
                    top_k = 0
                    top_p = 0.9
                    filtered_logits = []
                    for logit in logits:
                        filter_logit = top_k_top_p_filtering(logit, top_k=top_k, top_p=top_p)
                        filtered_logits.append(filter_logit)
                    filtered_logits = torch.stack(filtered_logits, dim=0)
                    # Sample from the filtered distribution
                    p = to_numpy(torch.softmax(filtered_logits, dim=-1))

                    # update generated sequence
                    for batch_index in range(batch_size):
                        new_pitch_index = np.random.choice(np.arange(
                            self.num_tokens_per_channel[channel_index]
                        ), p=p[batch_index])
                        x[batch_index, event_index, channel_index] = int(new_pitch_index)

                    # store attentions
                    if plot_attentions:
                        layer = 2
                        event_index_encoder = (event_index * self.num_channels) // self.total_upscaling
                        attentions_encoder = forward_pass['attentions_encoder']
                        # list of dicts with key 'a_self_encoder'
                        attentions_decoder = forward_pass['attentions_decoder']
                        # list of dicts with keys 'a_self_decoder' and 'a_cross'

                        # get attentions at corresponding event
                        attn_encoder = attentions_encoder[layer]['a_self_encoder'][:, :,
                                       event_index_encoder, :]
                        attn_decoder = attentions_decoder[layer]['a_self_decoder'][:, :,
                                       event_index * self.num_channels + channel_index, :]
                        attn_cross = attentions_decoder[layer]['a_cross'][:, :,
                                     event_index * self.num_channels + channel_index, :]

                        attentions_encoder_list.append(attn_encoder)
                        attentions_decoder_list.append(attn_decoder)
                        attentions_cross_list.append(attn_cross)

            # Compute codes for generations
            x_re_encode = torch.cat([
                cuda_variable(x_original_single.long()),
                x
            ], dim=0)
            _, recoding_, _ = self.encoder(x_re_encode)
            recoding_ = recoding_.detach().cpu().numpy()
            recoding = self.encoder.merge_codes(recoding_)

        # to score
        original_and_reconstruction = self.data_processor.postprocess(original=x_original.long(),
                                                                      reconstruction=x.cpu())

        ###############################
        # Saving
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        if not os.path.exists(f'{self.model_dir}/generations'):
            os.mkdir(f'{self.model_dir}/generations')

        # Write code sequence
        with open(f'{self.model_dir}/generations/{timestamp}.txt', 'w') as ff:
            for batch_ind in range(len(recoding)):
                aa = recoding[batch_ind]
                ff.write(' , '.join(map(str, list(aa))))
                ff.write('\n')

        # Write scores
        scores = []
        for k, tensor_score in enumerate(original_and_reconstruction):
            path_no_extension = f'{self.model_dir}/generations/{timestamp}_{k}'
            scores.append(self.dataloader_generator.write(tensor_score, path_no_extension))
        ###############################

        if plot_attentions:
            self.plot_attention(attentions_cross_list,
                                timestamp=timestamp,
                                name='attns_cross')
            self.plot_attention(attentions_encoder_list,
                                timestamp=timestamp,
                                name='self_attns_encoder')
            self.plot_attention(attentions_decoder_list,
                                timestamp=timestamp,
                                name='self_attns_decoder')

        return scores

    def init_generation(self, num_events):
        return cuda_variable(
            torch.zeros(1, num_events, self.num_channels).long()
        )

    def generate_from_code_long(self, encoding_indices, temperature,
                                num_decodings=1, code_index_start=None,
                                code_index_end=None):
        self.eval()
        size_encoding = encoding_indices.size(1)

        total_upscaling = int(np.prod(self.encoders_stack.downscale_factors))
        num_tokens_indices = self.data_processor.num_tokens // total_upscaling

        num_events_full_chorale = size_encoding * total_upscaling // self.data_processor.num_channels
        num_events_before_start = code_index_start * total_upscaling // self.num_channels
        num_events_before_end = code_index_end * total_upscaling // self.num_channels

        batch_size = num_decodings * encoding_indices.size(0)

        if code_index_start is None:
            code_index_start = 0
        if code_index_end is None:
            code_index_end = size_encoding

        with torch.no_grad():
            # TODO must be fixed for other datasets
            # chorale = self.init_generation(num_events=num_events_full_chorale)
            chorale = self.init_generation_chorale(num_events=num_events_full_chorale,
                                                   start_index=num_events_before_start)
            # Duplicate along batch dimension
            chorale = chorale.repeat(batch_size, 1, 1)
            encoding_indices = encoding_indices.repeat_interleave(num_decodings, dim=0)

            for code_index in range(code_index_start, code_index_end):
                for relative_event in range(self.num_events_per_code):
                    for channel_index in range(self.data_processor.num_channels):
                        t_begin, t_end, t_relative = self.compute_start_end_times(
                            code_index, num_blocks=size_encoding,
                            num_blocks_model=num_tokens_indices
                        )

                        input_encoding_indices = encoding_indices[:, t_begin:t_end]

                        input_chorale = chorale[:,
                            t_begin * self.num_events_per_code: t_end * self.num_events_per_code, :]
                        weights_per_voice = self.forward(input_encoding_indices,
                                                         input_chorale)['weights_per_category']
                        weights = weights_per_voice[channel_index]
                        probs = torch.softmax(
                            weights[:, t_relative * self.num_events_per_code + relative_event, :],
                            dim=1)
                        p = to_numpy(probs)
                        # temperature ?!
                        p = np.exp(np.log(p + 1e-20) * temperature)
                        p = p / p.sum(axis=1, keepdims=True)

                        for batch_index in range(batch_size):
                            new_pitch_index = np.random.choice(np.arange(
                                self.num_tokens_per_channel[channel_index]
                            ), p=p[batch_index])
                            chorale[batch_index,
                                    code_index * self.num_events_per_code + relative_event,
                                    channel_index] = int(
                                new_pitch_index)

        # slice
        chorale = chorale[:, num_events_before_start:num_events_before_end]
        tensor_score = self.data_processor.postprocess(original=None,
                                                       reconstruction=chorale)
        #  ToDo define path
        scores = self.dataloader_generator.write(tensor_score, path)
        return scores

    def compute_start_end_times(self, t, num_blocks, num_blocks_model):
        """

        :param t:
        :param num_blocks: num_blocks of the sequence to be generated
        :param num_blocks_model:
        :return:
        """
        # t_relative
        if num_blocks_model // 2 <= t < num_blocks - num_blocks_model // 2:
            t_relative = (num_blocks_model // 2)
        else:
            if t < num_blocks_model // 2:
                t_relative = t
            elif t >= num_blocks - num_blocks_model // 2:
                t_relative = num_blocks_model - (num_blocks - t)
            else:
                NotImplementedError

        # choose proper block to use
        t_begin = min(max(0, t - num_blocks_model // 2), num_blocks - num_blocks_model)
        t_end = t_begin + num_blocks_model

        return t_begin, t_end, t_relative

    def generate_reharmonisation(self, num_reharmonisations, temperature):
        """
        This method only works on bach chorales
        :param num_reharmonisations:
        :param temperature:
        :return:
        """
        import music21
        cl = music21.corpus.chorales.ChoraleList()
        print(cl.byBWV.keys())
        # chorale_m21 = music21.corpus.chorales.getByTitle(cl.byBWV[437]['title'])
        # chorale_m21 = music21.corpus.chorales.getByTitle(cl.byBWV[289]['title'])

        for bwv in cl.byBWV.keys():
            chorale_m21 = music21.corpus.chorales.getByTitle(cl.byBWV[bwv]['title'])
            x = self.dataloader_generator.dataset.transposed_score_and_metadata_tensors(
                chorale_m21, semi_tone=0)[0].transpose(1, 0).unsqueeze(0)
            # remove metadata
            # and put num_channels at the end
            # and add batch_dim

            x_chunks = list(x.split(self.data_processor.num_events, 1))

            last_chunk = x_chunks[-1]

            # compute START and END clusters
            PAD = [d[PAD_SYMBOL] for d in self.dataloader_generator.dataset.note2index_dicts]
            START = [d[START_SYMBOL] for d in self.dataloader_generator.dataset.note2index_dicts]
            END = [d[END_SYMBOL] for d in self.dataloader_generator.dataset.note2index_dicts]

            # start
            start_chunk_ = torch.Tensor(START).unsqueeze(0).unsqueeze(0).long()
            pad_chunk_beginning = torch.Tensor(PAD).unsqueeze(0).unsqueeze(0).repeat(
                1, self.data_processor.num_events - 1, 1
            ).long()
            start_chunk = torch.cat([pad_chunk_beginning, start_chunk_], 1)

            # end
            end_chunk_ = torch.Tensor(END).unsqueeze(0).unsqueeze(0).long()
            pad_chunk_end = torch.Tensor(PAD).unsqueeze(0).unsqueeze(0).repeat(
                1, self.data_processor.num_events - 1, 1
            ).long()
            end_chunk = torch.cat([end_chunk_, pad_chunk_end], 1)

            # last chunk
            completion_chunk = torch.Tensor(PAD).unsqueeze(0).unsqueeze(0).repeat(
                1, self.data_processor.num_events - last_chunk.size(1) - 1, 1
            ).long()
            last_chunk = torch.cat([last_chunk, end_chunk_, completion_chunk], 1)
            x_chunks[-1] = last_chunk

            x_chunks = torch.cat([start_chunk] + x_chunks + [end_chunk], dim=0)
            encoding_indices_stack = self.encoders_stack(x)
            encoding_indices = self.encoders_stack.merge_codes(encoding_indices_stack)
            print(encoding_indices.size())

            # glue all encoding indices
            encoding_indices = torch.cat(
                encoding_indices.split(1, 0),
                1
            )
            # compute code_index start and stop
            total_upscaling = int(np.prod(self.encoders_stack.downscale_factors))
            code_index_start = start_chunk.size(1) * self.num_channels // total_upscaling
            code_index_end = encoding_indices.size(1) - (
                    end_chunk.size(1) + completion_chunk.size(1)) * self.num_channels // \
                             total_upscaling

            scores = self.generate_from_code_long(encoding_indices,
                                                  num_decodings=num_reharmonisations,
                                                  temperature=temperature,
                                                  code_index_start=code_index_start,
                                                  code_index_end=code_index_end
                                                  )

            reharmonisation_dir = f'{self.model_dir}/reharmonisations'
            if not os.path.exists(reharmonisation_dir):
                os.mkdir(reharmonisation_dir)
            for k, score in enumerate(scores):
                score.write('xml', f'{reharmonisation_dir}/BWV{bwv}_{k}.xml')
                # score.show()
            chorale_m21.write('xml', f'{reharmonisation_dir}/BWV{bwv}_original.xml')
        return scores

    def generate_alla_mano(self, start_codes, end_codes, body_codes, temperature):
        code_index_start = len(start_codes)
        encoding_indices = start_codes + \
                           body_codes
        code_index_end = len(encoding_indices)
        encoding_indices += end_codes
        encoding_indices = torch.Tensor(encoding_indices).unsqueeze(0).long().to('cuda')

        scores = self.generate_from_code_long(
            encoding_indices=encoding_indices,
            temperature=temperature,
            num_decodings=3,
            code_index_start=code_index_start,
            code_index_end=code_index_end)

        save_dir = f'{self.model_dir}/alla_mano'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for k, score in enumerate(scores):
            score.write('xml', f'{save_dir}/{k}.xml')

        return scores

    def check_duplicate(self, generation, original):
        from difflib import SequenceMatcher
        s1 = self.data_processor.dump(generation)
        s2 = self.data_processor.dump(original)

        match = SequenceMatcher(None, s1, s2).find_longest_match(0, len(s1), 0, len(s2))
        print(match)
        print(s1[match.a: match.a + match.size])
        print(f'Num tokens plagiarisms: {(match.size - 1) / 3}')

    def check_duplicate_all_corpus(self, generation):
        from difflib import SequenceMatcher
        s1 = self.data_processor.dump(generation)
        (generator_train, generator_val, _) = self.dataloader_generator.dataloaders(
            batch_size=1,
            shuffle_val=True,
            shuffle_train=False
        )
        best_x = None
        best_size = 0
        for tensor_dict in tqdm(generator_train):
            x = tensor_dict['x']
            s2 = self.data_processor.dump(x[0])
            match = SequenceMatcher(None, s1, s2, autojunk=False).find_longest_match(0, len(s1),
                                                                                     0, len(s2))
            if match.size > best_size:
                best_x = x
                best_size = match.size
            # print(match)
            # print(s1[match.a: match.a + match.size])

        print(f'Num tokens plagiarisms: {(best_size - 1) / 3}')
        print(f'Num beats plagiarisms: {(best_size - 1) / 3 / 4 / 4}')

        return best_x

    def plot_attention(self,
                       attentions_list,
                       timestamp,
                       name):
        """
        Helper function

        :param attentions_list: list of (batch_size, num_heads, num_tokens_encoder

        :return:
        """
        # to (batch_size, num_heads, num_tokens_decoder, num_tokens_encoder)
        attentions_batch = torch.cat(
            [t.unsqueeze(2)
             for t in attentions_list
             ], dim=2
        )

        # plot only batch 0 for now
        for batch_index, attentions in enumerate(attentions_batch):
            plt.clf()
            plt.cla()
            num_heads = attentions.size(0)
            for head_index, t in enumerate(attentions):
                plt.subplot(1, num_heads, head_index + 1)
                plt.title(f'Head {head_index}')
                mat = t.detach().cpu().numpy()
                sns.heatmap(mat, vmin=0, vmax=1, cmap="YlGnBu")
                plt.grid(True)
            plt.savefig(f'{self.model_dir}/generations/{timestamp}_{batch_index}_{name}.pdf')
            # plt.show()
        plt.close()

    # TODO put this in data_processor/dataloader_generator
    # but hard!
    def init_generation_chorale(self, num_events, start_index):
        PAD = [d[PAD_SYMBOL] for d in self.dataloader_generator.dataset.note2index_dicts]
        START = [d[START_SYMBOL] for d in self.dataloader_generator.dataset.note2index_dicts]
        aa = torch.Tensor(PAD).unsqueeze(0).unsqueeze(0).repeat(1, start_index - 1, 1).long()
        bb = torch.Tensor(START).unsqueeze(0).unsqueeze(0).long()
        cc = torch.Tensor(PAD).unsqueeze(0).unsqueeze(0).repeat(1, num_events - start_index, 1).long()
        init_sequence = torch.cat([aa, bb, cc], 1)
        return init_sequence
