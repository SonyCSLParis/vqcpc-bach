import os
from itertools import islice

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from VQCPCB import vqcpc_helper
from VQCPCB.encoder import EncoderTrainer
from VQCPCB.vqcpc_helper import quantization_loss, nce_loss


class VQCPCEncoderTrainer(EncoderTrainer):
    def __init__(self,
                 model_dir,
                 dataloader_generator,
                 encoder,
                 c_net_kwargs,
                 quantization_weighting,
                 ):
        super(VQCPCEncoderTrainer, self).__init__(
            dataloader_generator=dataloader_generator
        )
        self.model_dir = model_dir
        self.dataloader_generator = dataloader_generator
        self.encoder = encoder
        self.data_processor = encoder.data_processor
        # compute information from encoder network
        self.codebook_dim = self.encoder.quantizer.codebook_dim
        self.upscale_factors = list(reversed(
            self.encoder.downscaler.downscale_factors
        )
        )
        self.num_tokens_per_channel = self.encoder.data_processor.num_tokens_per_channel
        self.num_channels = len(self.num_tokens_per_channel)

        assert self.data_processor.num_tokens % np.prod(self.upscale_factors) == 0
        # self.num_tokens_bottleneck = int(self.data_processor.num_tokens // np.prod(self.upscale_factors))

        #  Embeddings size
        if encoder.upscaler is not None:
            z_dim = encoder.upscaler.output_dim
        else:
            z_dim = self.codebook_dim
        c_dim = c_net_kwargs['output_dim']

        #  C net
        self.c_module = vqcpc_helper.CModule(
            input_dim=z_dim,
            hidden_size=c_net_kwargs['hidden_size'],
            output_dim=c_dim,
            num_layers=c_net_kwargs['num_layers'],
            dropout=c_net_kwargs['dropout']
        )

        self.fks_module = vqcpc_helper.FksModule(z_dim=z_dim,
                                                 c_dim=c_dim,
                                                 k_max=self.dataloader_generator.num_blocks_right,
                                                 )

        # optim
        self.quantization_weighting = quantization_weighting
        self.optimizer = None
        self.scheduler = None

    def init_optimizers(self, lr=1e-3):
        self.optimizer = torch.optim.Adam(
            list(self.c_module.parameters()) +
            list(self.fks_module.parameters()) +
            list(self.encoder.parameters()),
            lr=lr
        )

    def to(self, device):
        self.encoder.to(device)
        self.c_module.to(device)
        self.fks_module.to(device)

    def save(self, early_stopped):
        if early_stopped:
            model_dir = f'{self.model_dir}/early_stopped'
        else:
            model_dir = f'{self.model_dir}/overfitted'

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.encoder.save(early_stopped=early_stopped)
        torch.save(self.c_module.state_dict(), f'{model_dir}/c_module')
        torch.save(self.fks_module.state_dict(), f'{model_dir}/fks_module')
        # print(f'Model {self.__repr__()} saved')

    def load(self, early_stopped):
        print(f'Loading models {self.__repr__()}')
        if early_stopped:
            model_dir = f'{self.model_dir}/early_stopped'
        else:
            model_dir = f'{self.model_dir}/overfitted'

        #  Deal with older configs
        if not os.path.exists(model_dir):
            model_dir = self.model_dir

        self.encoder.load(early_stopped=early_stopped)
        self.c_module.load_state_dict(torch.load(f'{model_dir}/c_module'))
        self.fks_module.load_state_dict(torch.load(f'{model_dir}/fks_module'))

    def train(self):
        self.encoder.train()
        self.c_module.train()
        self.fks_module.train()

    def eval(self):
        self.encoder.eval()
        self.c_module.eval()
        self.fks_module.eval()

    def epoch(self,
              data_loader,
              train,
              num_batches,
              corrupt_labels,
              ):

        means = {
            'loss': 0,
            'accuracy': 0,
            'loss_quantize': 0,
            'loss_contrastive': 0,
            'num_codewords': 0,
            'num_codewords_negative': 0,
        }

        print(f'lr: {self.optimizer.param_groups[0]["lr"]}')

        if train:
            self.train()
        else:
            self.eval()

        for sample_id, tensor_dict in tqdm(enumerate(islice(data_loader,
                                                            num_batches))):

            continue

            # downscale and quantize (preprocessing and embedding are included in these steps)
            negative_samples = tensor_dict['negative_samples']
            batch_size, num_negative_samples, fks_dim, num_events, num_channels = negative_samples.shape
            negative_samples = negative_samples.view(batch_size * num_negative_samples * fks_dim, num_events,
                                                     num_channels)
            z_quantized_negative, encoding_indices_negative, quantization_loss_negative = self.encoder(
                negative_samples, corrupt_labels=corrupt_labels)
            _, num_blocks, dim_z = z_quantized_negative.shape
            z_quantized_negative = z_quantized_negative.view(batch_size, num_negative_samples, fks_dim, num_blocks,
                                                             dim_z)
            if encoding_indices_negative is not None:
                encoding_indices_negative = encoding_indices_negative.view(batch_size, num_negative_samples, fks_dim,
                                                                           num_blocks)

            quantization_loss_negative = quantization_loss_negative.view(batch_size, num_negative_samples, fks_dim,
                                                                         num_blocks)

            z_quantized_left, encoding_indices_left, quantization_loss_left = self.encoder(tensor_dict['x_left'],
                                                                                           corrupt_labels=False)
            z_quantized_right, encoding_indices_right, quantization_loss_right = self.encoder(tensor_dict['x_right'],
                                                                                              corrupt_labels=False)
            # -- compute c
            c = self.c_module(z_quantized_left, h=None)

            #  -- Positive fks
            fks_positive = self.fks_module(c, z_quantized_right)

            #  --Negative fks
            # z_negative is
            # (batch_size, num_negative_samples, num_blocks_right, 1, z_dim)
            z_quantized_negative = z_quantized_negative[:, :, :, 0, :]
            (batch_size,
             num_negative_samples,
             num_blocks_right,
             z_dim) = z_quantized_negative.size()

            z_quantized_negative = z_quantized_negative.permute(1, 0, 2, 3).contiguous().view(
                batch_size * num_negative_samples,
                num_blocks_right,
                z_dim
            )

            c_repeat = c.repeat(num_negative_samples, 1)
            fks_negative = self.fks_module(c_repeat, z_quantized_negative)

            fks_negative = fks_negative.view(num_negative_samples,
                                             batch_size,
                                             num_blocks_right
                                             ).contiguous().permute(1, 2, 0)

            # fks_negative is now (batch_size, k, num_negative_examples)
            # fks_positive is (batch_size, k)

            # -- compute score:
            score_matrix = fks_positive > fks_negative.max(2)[0]
            #########################

            # == Compute loss
            # -- contrastive loss
            contrastive_loss = nce_loss(fks_positive, fks_negative)

            q_loss = quantization_loss(quantization_loss_left,
                                       quantization_loss_negative,
                                       quantization_loss_right)

            loss = contrastive_loss + self.quantization_weighting * q_loss

            # == Optim
            self.optimizer.zero_grad()
            if train:
                loss.backward()
                nn.utils.clip_grad_value_(self.parameters(), clip_value=5)
                self.optimizer.step()
            #########################

            # Monitored quantities and clean
            means['loss'] += loss.item()
            means['loss_quantize'] += q_loss.item()
            means['loss_contrastive'] += contrastive_loss.item()

            # compute num codewords
            if encoding_indices_left is not None:
                means['num_codewords'] += len(torch.unique(
                    torch.cat((encoding_indices_left, encoding_indices_right), dim=0)
                ))
                means['num_codewords_negative'] += len(torch.unique(
                    encoding_indices_negative
                ))

            del contrastive_loss
            del q_loss
            del loss

            accuracy = score_matrix.sum(dim=0).float() / batch_size
            means['accuracy'] += accuracy.detach().cpu().numpy()
            del accuracy

        norm_effective_num_batches = sample_id + 1
        # Re-normalize monitored quantities and free gpu memory
        means = {
            key: (value / norm_effective_num_batches)
            for key, value in means.items()
        }

        means['accuracy'] = list(means['accuracy'])
        #  Use mean accuracy as monitor loss
        means['loss_monitor'] = - sum(means['accuracy']) / len(means['accuracy'])

        return means