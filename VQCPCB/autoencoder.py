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

from VQCPCB.data.chorale_dataset import END_SYMBOL, START_SYMBOL, PAD_SYMBOL
from VQCPCB.data.data_processor import DataProcessor
from VQCPCB.decoder import Decoder
from VQCPCB.encoder import Encoder
from VQCPCB.utils import cuda_variable, categorical_crossentropy, flatten, dict_pretty_print, \
    top_k_top_p_filtering, \
    to_numpy


class Autoencoder(nn.Module):
    def __init__(self,
                 model_dir,
                 dataloader_generator,
                 data_processor: DataProcessor,
                 encoder,
                 decoder):
        """
        Like DecoderCustom, but the positioning is relative
        :param model_dir:
        :param dataloader_generator:
        :param data_processor:
        :param encoder:
        """
        super(Autoencoder, self).__init__()

        # encoder
        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder
        self.model_dir = model_dir
        self.dataloader_generator = dataloader_generator
        self.data_processor = data_processor
        self.optimizer = None
        self.scheduler = None

    def __repr__(self):
        return f'Autoencoder'

    def init_optimizers(self, lr, schedule_lr):
        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.parameters()),
            lr=lr
        )
        # Scheduler
        if schedule_lr:
            warmup_steps = 10000
            # lr will evolved between min_scaling * lr and max_scaling *lr (up and down more slowly)
            min_scaling = 0.1
            max_scaling = 1
            slope_1 = (max_scaling - min_scaling) / warmup_steps
            slope_2 = - slope_1 * 0.1
            lr_schedule = \
                lambda epoch: max(min(min_scaling + slope_1 * epoch,
                                      max_scaling + (epoch - warmup_steps) * slope_2),
                                  min_scaling
                                  )
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_schedule)

    def save(self, early_stopped):
        if early_stopped:
            model_dir = f'{self.model_dir}/early_stopped'
        else:
            model_dir = f'{self.model_dir}/overfitted'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.encoder.save(early_stopped)
        self.decoder.save(early_stopped)

    def load(self, early_stopped, device):
        self.encoder.load(early_stopped, device)
        self.decoder.load(early_stopped, device)

    def epoch(self, data_loader,
              train=True,
              num_batches=None,
              ):
        means = None

        if train:
            self.train()
        else:
            self.eval()

        for sample_id, tensor_dict in tqdm(enumerate(
                islice(data_loader, num_batches)),
                ncols=80):

            self.optimizer.zero_grad()
            x = tensor_dict['x']
            forward_pass = self.forward(x)
            loss = forward_pass['loss']
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
                self.optimizer.step()
            if train and (self.scheduler is not None):
                self.scheduler.step()

            # Monitored quantities
            monitored_quantities = forward_pass['monitored_quantities']

            # average quantities
            if means is None:
                means = {key: 0
                         for key in monitored_quantities if key != 'codebook_usage_histo'}
                means['codebook_usage_histo'] = np.zeros_like(monitored_quantities['codebook_usage_histo'])
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
        means['codebook_usage_histo'] = list(means['codebook_usage_histo'])
        return means

    def train_model(self,
                    batch_size,
                    num_batches,
                    num_epochs,
                    lr,
                    schedule_lr,
                    plot=False,
                    num_workers=0,
                    **kwargs
                    ):
        if plot:
            self.writer = SummaryWriter(f'{self.model_dir}')

        best_val = 1e8
        self.init_optimizers(lr=lr, schedule_lr=schedule_lr)
        for epoch_id in range(num_epochs):
            (generator_train,
             generator_val,
             generator_test) = self.dataloader_generator.dataloaders(
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle_train=True,
                shuffle_val=True)

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

    def forward(self, x):

        """
        :param source: sequence of codebooks (batch_size, s_s)
        :param target: sequence of tokens (batch_size, num_events, num_channels)
        :return:
        """
        # compute encoding_indices version
        z_quantized, encoding_indices, quantization_loss = self.encoder(x)
        q_loss = quantization_loss.mean()
        # compute num codewords
        codebook_usage_histo, _ = np.histogram(np.asarray(encoding_indices.tolist()),
                                               bins=self.encoder.quantizer.codebook_size,
                                               range=(0, 8))

        # compute encoding_indices version
        decoder_output = self.decoder(source=z_quantized,
                                      target=x)
        reconstruction_loss = decoder_output['loss']
        loss = reconstruction_loss + q_loss
        return {
            'loss': loss,
            'attentions_decoder': decoder_output['attentions_decoder'],
            'attentions_encoder': decoder_output['attentions_encoder'],
            'weights_per_category': decoder_output['weights_per_category'],
            'monitored_quantities': {
                'loss': loss.item(),
                'reconstruction_loss': reconstruction_loss.item(),
                'quantization_loss': q_loss.item(),
                'codebook_usage_histo': codebook_usage_histo
            }
        }

    def decode(self, source, target):
        batch_size = source.size(0)
        # embed
        source_seq = self.source_embeddings(source)

        target = cuda_variable(target.long())
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
            memory_mask = self._generate_anticausal_mask(source_length, target_length)

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
        reconstruction_loss = categorical_crossentropy(
            value=weights_per_category,
            target=target,
            mask=torch.ones_like(target)
        )
        return reconstruction_loss

    def plot(self, epoch_id, monitored_quantities_train,
             monitored_quantities_val):
        if monitored_quantities_train is not None:
            for k, v in monitored_quantities_train.items():
                if k == 'codebook_usage_histo':
                    fig = plt.figure()
                    y = np.asarray(v)
                    plt.bar(x=np.arange(y.shape[0]),
                            height=y
                            )
                    self.writer.add_figure(f'codebook_{k}_usage', fig,
                                           global_step=epoch_id,
                                           close=True)
                else:
                    self.writer.add_scalar(f'{k}/train', v, epoch_id)

        if monitored_quantities_val is not None:
            for k, v in monitored_quantities_val.items():
                if k == 'codebook_usage_histo':
                    fig = plt.figure()
                    y = np.asarray(v)
                    plt.bar(x=np.arange(y.shape[0]),
                            height=y
                            )
                    self.writer.add_figure(f'codebook_{k}_usage', fig,
                                           global_step=epoch_id,
                                           close=True)
                else:
                    self.writer.add_scalar(f'{k}/val', v, epoch_id)

    def generate(self, temperature,
                 batch_size=1,
                 top_k=0,
                 top_p=1.,
                 seed_set=None,
                 exclude_meta_symbols=False,
                 plot_attentions=False,
                 code_juxtaposition=False):
        self.eval()
        (generator_train, generator_val, _) = self.dataloader_generator.dataloaders(
            batch_size=1,
            num_workers=0,
            shuffle_train=True,
            shuffle_val=True
        )

        with torch.no_grad():
            if code_juxtaposition:
                # Use the codes of a chorale for the first half, and the codes from another chorale for the last half
                if seed_set == 'val':
                    tensor_dict_beginning = next(iter(generator_val))
                    tensor_dict_end = next(iter(generator_val))
                elif seed_set == 'train':
                    tensor_dict_beginning = next(iter(generator_train))
                    tensor_dict_end = next(iter(generator_train))
                else:
                    raise Exception('Need to indicate seeds dataset')

                num_events_chorale_half = tensor_dict_beginning['x'].shape[1] // 2
                x_beg = tensor_dict_beginning['x'][:, :num_events_chorale_half]
                x_end = tensor_dict_end['x'][:, num_events_chorale_half:]
                x_original_single = torch.cat([x_beg, x_end], dim=1)
                x_original = x_original_single.repeat(batch_size, 1, 1)
            else:
                if seed_set == 'val':
                    tensor_dict = next(iter(generator_val))
                elif seed_set == 'train':
                    tensor_dict = next(iter(generator_train))
                else:
                    raise Exception('Need to indicate seeds dataset')

                x_original_single = tensor_dict['x']
                x_original = x_original_single.repeat(batch_size, 1, 1)

            # compute downscaled version
            zs, encoding_indices, _ = self.encoder(x_original)
            if encoding_indices is None:
                # if no quantization is used, directly use the zs
                encoding_indices = zs

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
                    if exclude_meta_symbols:
                        for sym in [START_SYMBOL, END_SYMBOL, PAD_SYMBOL]:
                            sym_index = \
                                self.dataloader_generator.dataset.note2index_dicts[channel_index][
                                    sym]
                            logits[:, sym_index] = -float("inf")

                    # Top-p sampling
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
                        event_index_encoder = (
                                                      event_index * self.num_channels) // self.total_upscaling
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
            _, recoding, _ = self.encoder(x_re_encode)
            if recoding is not None:
                recoding = recoding.detach().cpu().numpy()

        # to score
        original_and_reconstruction = self.data_processor.postprocess(original=x_original.long(),
                                                                      reconstruction=x.cpu())

        ###############################
        # Saving
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if code_juxtaposition:
            save_dir = f'{self.model_dir}/juxtapositions'
        else:
            save_dir = f'{self.model_dir}/generations'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # Write code sequence
        if recoding is not None:
            with open(f'{save_dir}/{timestamp}.txt', 'w') as ff:
                for batch_ind in range(len(recoding)):
                    aa = recoding[batch_ind]
                    ff.write(' , '.join(map(str, list(aa))))
                    ff.write('\n')

        # Write scores
        scores = []
        for k, tensor_score in enumerate(original_and_reconstruction):
            path_no_extension = f'{save_dir}/{timestamp}_{k}'
            scores.append(self.dataloader_generator.write(tensor_score, path_no_extension))
        print(f'Saved in {save_dir}/{timestamp}')
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

    def generate_from_code_long(self, encoding_indices,
                                temperature,
                                top_k=0,
                                top_p=1.,
                                exclude_meta_symbols=False,
                                num_decodings=1,
                                code_index_start=None,
                                code_index_end=None):
        """
        Returns a list of music21 scores
        """
        self.eval()
        size_encoding = encoding_indices.size(1)

        total_upscaling = int(np.prod(self.encoder.downscaler.downscale_factors))
        win_size_codes = self.data_processor.num_tokens // total_upscaling
        assert win_size_codes % 4 == 0
        hop_size_codes = win_size_codes // 4

        num_events_full_chorale = size_encoding * total_upscaling // self.data_processor.num_channels
        num_events_before_start = code_index_start * total_upscaling // self.num_channels
        num_events_before_end = code_index_end * total_upscaling // self.num_channels

        batch_size = num_decodings * encoding_indices.size(0)

        if code_index_start is None:
            code_index_start = 0
        if code_index_end is None:
            code_index_end = size_encoding

        with torch.no_grad():
            chorale = self.init_generation_chorale(num_events=num_events_full_chorale,
                                                   start_index=num_events_before_start)
            # Duplicate along batch dimension
            chorale = chorale.repeat(batch_size, 1, 1)
            encoding_indices = encoding_indices.repeat_interleave(num_decodings, dim=0)

            for code_index in range(code_index_start, code_index_end):
                for relative_event in range(self.num_events_per_code):
                    for channel_index in range(self.data_processor.num_channels):
                        ####################################
                        # Codes indices for extracting the inputs
                        middle_code_ind_segment = (code_index // hop_size_codes) * hop_size_codes
                        start_code_ind_segment = middle_code_ind_segment - win_size_codes // 2
                        end_code_ind_segment = middle_code_ind_segment + win_size_codes // 2
                        event_start_segment = start_code_ind_segment * self.num_events_per_code
                        event_end_segment = end_code_ind_segment * self.num_events_per_code

                        # Where do we sample in this segment
                        event_sample = (code_index - start_code_ind_segment) * self.num_events_per_code + relative_event

                        ####################################
                        # Prepare network inputs (segments)
                        input_encoding_indices = encoding_indices[:, start_code_ind_segment:end_code_ind_segment]
                        input_chorale = chorale[:, event_start_segment: event_end_segment, :]

                        ####################################
                        # Forward pass in decoder
                        weights_per_voice = self.forward(input_encoding_indices,
                                                         input_chorale)['weights_per_category']
                        # Keep only the last token predictions of the first batch item (batch size 1)
                        weights = weights_per_voice[channel_index]
                        logits = weights[:, event_sample, :] / temperature

                        ####################################
                        # Remove meta symbols
                        if exclude_meta_symbols:
                            for sym in [START_SYMBOL, END_SYMBOL, PAD_SYMBOL]:
                                sym_index = \
                                    self.dataloader_generator.dataset.note2index_dicts[
                                        channel_index][
                                        sym]
                                logits[:, sym_index] = -float("inf")

                        ####################################
                        # Top-p sampling
                        filtered_logits = []
                        for logit in logits:
                            filter_logit = top_k_top_p_filtering(logit, top_k=top_k, top_p=top_p)
                            filtered_logits.append(filter_logit)
                        filtered_logits = torch.stack(filtered_logits, dim=0)
                        # Sample from the filtered distribution
                        p = to_numpy(torch.softmax(filtered_logits, dim=-1))

                        ####################################
                        # Sample and add to generated score
                        for batch_index in range(batch_size):
                            new_pitch_index = np.random.choice(np.arange(
                                self.num_tokens_per_channel[channel_index]
                            ), p=p[batch_index])
                            chorale[batch_index,
                                    code_index * self.num_events_per_code + relative_event,
                                    channel_index] = int(new_pitch_index)

        # slice
        chorale = chorale[:, num_events_before_start:num_events_before_end]
        tensor_scores = to_numpy(chorale)
        # Write scores
        scores = []
        for k, tensor_score in enumerate(tensor_scores):
            scores.append(self.dataloader_generator.to_score(tensor_score))
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
                raise Exception

        # choose proper block to use
        t_begin = min(max(0, t - num_blocks_model // 2), num_blocks - num_blocks_model)
        t_end = t_begin + num_blocks_model

        return t_begin, t_end, t_relative

    def generate_reharmonisation(self,
                                 num_reharmonisations,
                                 temperature,
                                 top_k,
                                 top_p):
        """
        This method only works on bach chorales
        :param num_reharmonisations:
        :param temperature:
        :return:
        """
        import music21
        cl = music21.corpus.chorales.ChoraleList()
        print(f'# Chorale BWV\n{list(cl.byBWV.keys())}')

        for bwv in cl.byBWV.keys():
            if bwv != 251:
                continue
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
            end_pad_chunk = torch.cat([end_chunk_, pad_chunk_end], 1)
            pad_only_chunk = torch.Tensor(PAD).unsqueeze(0).unsqueeze(0).repeat(
                1, self.data_processor.num_events, 1
            ).long()
            completion_length = self.data_processor.num_events - last_chunk.size(1)
            if completion_length > 1:
                pad_completion_chunk = torch.Tensor(PAD).unsqueeze(0).unsqueeze(0).repeat(
                    1, completion_length - 1, 1
                ).long()
                last_chunk = torch.cat([last_chunk, end_chunk_, pad_completion_chunk], 1)
                end_chunk = pad_only_chunk
            elif completion_length == 1:
                last_chunk = torch.cat([last_chunk, end_chunk_], 1)
                end_chunk = pad_only_chunk
            elif completion_length == 0:
                last_chunk = last_chunk
                end_chunk = end_pad_chunk
            x_chunks[-1] = last_chunk
            x_chunks = torch.cat([start_chunk] + x_chunks + [end_chunk], dim=0)

            zs, encoding_indices, _ = self.encoder(x_chunks)
            if encoding_indices is None:
                # if no quantization is used, directly use the zs
                encoding_indices = zs
            print(encoding_indices.size())

            # glue all encoding indices
            encoding_indices = torch.cat(
                encoding_indices.split(1, 0),
                1
            )
            # compute code_index start and stop
            total_upscaling = int(np.prod(self.encoder.downscaler.downscale_factors))
            code_index_start = start_chunk.size(1) * self.num_channels // total_upscaling
            code_index_end = encoding_indices.size(1) - \
                             (end_chunk.size(1) + completion_length) \
                             * self.num_channels // total_upscaling

            scores = self.generate_from_code_long(encoding_indices,
                                                  num_decodings=num_reharmonisations,
                                                  temperature=temperature,
                                                  top_k=top_k,
                                                  top_p=top_p,
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

    def init_generation_chorale(self, num_events, start_index):
        PAD = [d[PAD_SYMBOL] for d in self.dataloader_generator.dataset.note2index_dicts]
        START = [d[START_SYMBOL] for d in self.dataloader_generator.dataset.note2index_dicts]
        aa = torch.Tensor(PAD).unsqueeze(0).unsqueeze(0).repeat(1, start_index - 1, 1).long()
        bb = torch.Tensor(START).unsqueeze(0).unsqueeze(0).long()
        cc = torch.Tensor(PAD).unsqueeze(0).unsqueeze(0).repeat(1, num_events - start_index,
                                                                1).long()
        init_sequence = torch.cat([aa, bb, cc], 1)
        return init_sequence
