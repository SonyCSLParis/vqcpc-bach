import os
from datetime import datetime
from itertools import islice

import numpy as np
import torch
from torch import nn
from torch.nn.modules import Transformer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from VQCPCB.data_processor.data_processor import DataProcessor
from VQCPCB.dataloaders.dataloader_generator import DataloaderGenerator
from VQCPCB.utils import dict_pretty_print, cuda_variable, categorical_crossentropy, flatten, \
    to_numpy


class Decoder(nn.Module):
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
        super(Decoder, self).__init__()
        self.model_dir = model_dir
        self.encoder = encoder
        # freeze encoder
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.dataloader_generator = dataloader_generator
        self.data_processor = data_processor

        self.num_tokens_per_channel = self.data_processor.num_tokens_per_channel
        self.num_channels = len(self.num_tokens_per_channel)
        self.d_model = d_model
        # Compute num_tokens for source and target
        num_tokens_target = self.data_processor.num_tokens
        assert num_tokens_target % np.prod(self.encoder.downscaler.downscale_factors) == 0
        num_tokens_source = num_tokens_target // np.prod(self.encoder.downscaler.downscale_factors)

        self.source_positional_embeddings = nn.Parameter(
            torch.randn((1,
                         num_tokens_source,
                         positional_embedding_size))
        )

        self.target_positional_embeddings = nn.Parameter(
            torch.randn((1,
                         num_tokens_target,
                         positional_embedding_size))
        )

        self.embedding_dim = self.d_model - positional_embedding_size

        # TODO factorised positional embeddings
        # TODO put the whole model in DataParallel
        self.codebook_size = self.encoder.quantizer.codebook_size ** \
                             self.encoder.quantizer.num_codebooks
        self.source_embeddings = nn.Embedding(
            self.codebook_size, self.embedding_dim
        )

        #  Transformer
        self.transformer = Transformer(
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

    def init_optimizers(self, lr=1e-3):
        # TODO use radam
        self.optimizer = torch.optim.Adam(
            list(self.parameters())
            ,
            lr=lr
        )

    def __repr__(self):
        return 'Decoder'

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

            # ======== Get codes from Encoder ==================
            with torch.no_grad():
                x = tensor_dict['x']
                # compute encoding_indices version
                z_quantized, encoding_indices, quantization_loss = self.encoder(x)

            # ======== Train decoder =============
            self.optimizer.zero_grad()
            forward_pass = self.forward(
                encoding_indices,
                x
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

    def generate(self, temperature, batch_size=1):
        self.eval()
        (generator_train, generator_val, _) = self.dataloader_generator.dataloaders(
            batch_size=1,
            shuffle_val=True
        )
        with torch.no_grad():
            tensor_dict = next(iter(generator_val))
            # tensor_dict = next(iter(generator_train))

            x_original = tensor_dict['x']
            x_original = x_original.repeat(batch_size, 1, 1)
            # compute downscaled version
            (z_quantized,
             encoding_indices,
             quantization_loss) = self.encoder(x_original)

            x = self.init_generation(num_events=self.data_processor.num_events)

            # Duplicate along batch dimension
            x = x.repeat(batch_size, 1, 1)

            # TODO are these symbols used in all datasets?
            exclude_symbols = ['START', 'END']

            for event_index in range(self.data_processor.num_events):
                for channel_index in range(self.num_channels):
                    weights_per_voice = self.forward(encoding_indices,
                                                     x)['weights_per_category']

                    weights = weights_per_voice[channel_index]
                    probs = torch.softmax(
                        weights[:, event_index, :],
                        dim=1)
                    p = to_numpy(probs)
                    # temperature ?!
                    p = np.exp(np.log(p + 1e-20) * temperature)

                    # TODO maybe remove this (the original x can have start symbols)
                    # Removing these lines make the method applicable to all datasets
                    # exclude non note symbols:
                    # for sym in exclude_symbols:
                    #     sym_index = self.dataset.note2index_dicts[channel_index][sym]
                    #     p[:, sym_index] = 0
                    p = p / p.sum(axis=1, keepdims=True)

                    for batch_index in range(batch_size):
                        new_pitch_index = np.random.choice(np.arange(
                            self.num_tokens_per_channel[channel_index]
                        ), p=p[batch_index])
                        x[batch_index, event_index, channel_index] = int(new_pitch_index)

        original_and_reconstruction = torch.cat([
            x_original.long(),
            x.cpu()
        ], dim=1)

        # to score
        original_and_reconstruction = self.data_processor.postprocess(original=x_original.long(),
                                                                      reconstruction=x.cpu())
        scores = self.dataloader_generator.to_score(original_and_reconstruction)

        # save scores in model_dir
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        if not os.path.exists(f'{self.model_dir}/generations'):
            os.mkdir(f'{self.model_dir}/generations')

        for k, score in enumerate(scores):
            score.write('xml', f'{self.model_dir}/generations/{timestamp}_{k}.xml')

        return scores

    def init_generation(self, num_events):
        return cuda_variable(
            torch.zeros(1, num_events, self.num_channels).long()
        )

    def generate_from_code_long(self, encoding_indices, temperature, batch_size=1):
        self.eval()
        size_encoding = encoding_indices.size(1)

        num_tokens_indices = int(
            self.data_processor.num_tokens // np.prod(
                self.encoder.downscaler.downscale_factors)
        )
        num_events_full_chorale = int(
            size_encoding * np.prod(
                self.encoder.downscaler.downscale_factors)) // self.data_processor.num_channels

        with torch.no_grad():
            chorale = self.init_generation(num_events=num_events_full_chorale
                                           )
            # Duplicate along batch dimension
            chorale = chorale.repeat(batch_size, 1, 1)
            encoding_indices = encoding_indices.repeat(batch_size, 1)

            for code_index in range(size_encoding):
                for relative_event in range(self.num_events_per_code):
                    for channel_index in range(self.data_processor.num_channels):
                        t_begin, t_end, t_relative = self.compute_start_end_times(
                            code_index, num_blocks=size_encoding,
                            num_blocks_model=num_tokens_indices
                        )

                        input_encoding_indices = encoding_indices[:, t_begin:t_end]
                        # todo hardcoded
                        input_chorale = chorale[:, t_begin * self.num_events_per_code: t_end * self.num_events_per_code, :]
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

        tensor_score = self.data_processor.postprocess(original=None,
                                                       reconstruction=chorale)
        # ToDo define path
        scores = self.dataloader_generator.to_score(tensor_score, path)
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

    def generate_reharmonisation(self, batch_size, temperature):
        """
        This method only works on bach chorales
        :param batch_size:
        :param temperature:
        :return:
        """
        import music21
        cl = music21.corpus.chorales.ChoraleList()
        print(cl.byBWV.keys())
        # chorale_m21 = music21.corpus.chorales.getByTitle(cl.byBWV[437]['title'])
        # chorale_m21 = music21.corpus.chorales.getByTitle(cl.byBWV[289]['title'])
        for bwv in cl.byBWV.keys():
            try:
                chorale_m21 = music21.corpus.chorales.getByTitle(cl.byBWV[bwv]['title'])
                x = self.dataloader_generator.dataset.transposed_score_and_metadata_tensors(
                    chorale_m21, semi_tone=0)[0].transpose(1, 0).unsqueeze(0)
                # remove metadata
                # and put num_channels at the end
                # and add batch_dim

                raise Exception("Il manque les start end symbols et tout la, cf decoder_relative.py")
                x_chunks = list(x.split(self.data_processor.num_events, 1))

                last_chunk = x_chunks[-1]

                if last_chunk.size(1) < self.data_processor.num_events:
                    first_chunk = x_chunks[0]
                    last_chunk = torch.cat([
                        last_chunk,
                        first_chunk[:, :self.data_processor.num_events - last_chunk.size(1)]
                    ], dim=1)
                    x_chunks[-1] = last_chunk

                x_chunks = torch.cat(x_chunks, dim=0)
                z_quantized, encoding_indices, quantization_loss = self.encoder(x_chunks)
                print(encoding_indices.size())

                # glue all encoding indices
                encoding_indices = torch.cat(
                    encoding_indices.split(1, 0),
                    1
                )
                scores = self.generate_from_code_long(encoding_indices,
                                                      temperature=temperature)

                reharmonisation_dir = f'{self.model_dir}/reharmonisations'
                if not os.path.exists(reharmonisation_dir):
                    os.mkdir(reharmonisation_dir)
                for k, score in enumerate(scores):
                    score.write('xml', f'{reharmonisation_dir}/BWV{bwv}_{k}.xml')
                    # score.show()
                chorale_m21.write('xml', f'{reharmonisation_dir}/BWV{bwv}_original.xml')
            except:
                pass

        return scores
