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

        if c_net_kwargs['bidirectional']:
            self.c_module_back = vqcpc_helper.CModule(
                input_dim=z_dim,
                hidden_size=c_net_kwargs['hidden_size'],
                output_dim=c_dim,
                num_layers=c_net_kwargs['num_layers'],
                dropout=c_net_kwargs['dropout']
            )

            self.fks_module_back = vqcpc_helper.FksModule(z_dim=z_dim,
                                                          c_dim=c_dim,
                                                          k_max=self.dataloader_generator.num_blocks_right,
                                                          )
        else:
            self.c_module_back = None

        # optim
        self.quantization_weighting = quantization_weighting
        self.optimizer = None
        self.scheduler = None

    def init_optimizers(self, lr, schedule_lr):
        # Optimizer
        list_parameters = list(self.c_module.parameters()) \
                          + list(self.fks_module.parameters()) \
                          + list(self.encoder.parameters())
        if self.c_module_back is not None:
            list_parameters = list_parameters \
                              + list(self.fks_module_back.parameters()) \
                              + list(self.c_module_back.parameters())
        self.optimizer = torch.optim.Adam(list_parameters, lr=lr)

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

    def to(self, device):
        self.encoder.to(device)
        self.c_module.to(device)
        self.fks_module.to(device)
        if self.c_module_back is not None:
            self.c_module_back.to(device)
            self.fks_module_back.to(device)

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
        if self.c_module_back is not None:
            torch.save(self.c_module_state.state_dict(), f'{model_dir}/c_module_back')
            torch.save(self.fks_module_state.state_dict(), f'{model_dir}/fks_module_back')

    def load(self, early_stopped, device):
        print(f'Loading models {self.__repr__()}')
        if early_stopped:
            model_dir = f'{self.model_dir}/early_stopped'
        else:
            model_dir = f'{self.model_dir}/overfitted'

        #  Deal with older configs
        if not os.path.exists(model_dir):
            model_dir = self.model_dir

        self.encoder.load(early_stopped=early_stopped, device=device)
        self.c_module.load_state_dict(torch.load(f'{model_dir}/c_module', map_location=torch.device(device)))
        self.fks_module.load_state_dict(torch.load(f'{model_dir}/fks_module', map_location=torch.device(device)))
        if self.c_module_back is not None:
            self.c_module_back.load_state_dict(
                torch.load(f'{model_dir}/c_module_back', map_location=torch.device(device)))
            self.fks_module_back.load_state_dict(
                torch.load(f'{model_dir}/fks_module_back', map_location=torch.device(device)))

    def train(self):
        self.encoder.train()
        self.c_module.train()
        self.fks_module.train()
        if self.c_module_back is not None:
            self.c_module_back.train()
            self.fks_module_back.train()

    def eval(self):
        self.encoder.eval()
        self.c_module.eval()
        self.fks_module.eval()
        if self.c_module_back is not None:
            self.c_module_back.eval()
            self.fks_module_back.eval()

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
            ############################################################
            # ENCODE (downscale and quantize)
            # negatives
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
            if self.c_module_back is not None:
                negative_samples_back = tensor_dict['negative_samples_back']
                negative_samples_back = negative_samples_back.view(batch_size * num_negative_samples * fks_dim,
                                                                   num_events, num_channels)
                z_quantized_negative_back, encoding_indices_negative_back, quantization_loss_negative_back = \
                    self.encoder(negative_samples_back, corrupt_labels=corrupt_labels)
                z_quantized_negative_back = z_quantized_negative_back.view(batch_size, num_negative_samples, fks_dim,
                                                                           num_blocks, dim_z)
                if encoding_indices_negative_back is not None:
                    encoding_indices_negative_back = encoding_indices_negative_back.view(batch_size,
                                                                                         num_negative_samples,
                                                                                         fks_dim,
                                                                                         num_blocks)
                quantization_loss_negative_back = quantization_loss_negative_back.view(batch_size, num_negative_samples,
                                                                                       fks_dim, num_blocks)
            # left positives
            z_quantized_left, encoding_indices_left, quantization_loss_left = self.encoder(tensor_dict['x_left'],
                                                                                           corrupt_labels=False)
            # right positives
            z_quantized_right, encoding_indices_right, quantization_loss_right = self.encoder(tensor_dict['x_right'],
                                                                                              corrupt_labels=False)
            ############################################################

            ############################################################
            #  COMPUTE FORWARD PREDICTIONS
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

            # -- contrastive loss
            contrastive_loss = nce_loss(fks_positive, fks_negative)
            ############################################################

            ############################################################
            #  fks backward
            if self.c_module_back is not None:
                z_quantized_right_flip = z_quantized_right.flip(dims=[1])
                c_back = self.c_module_back(z_quantized_right_flip, h=None)

                #  -- Positive fks
                # DO NOT FLIP LEFT Zs as they were not flipped to construct the negatives samples
                fks_positive_back = self.fks_module_back(c_back, z_quantized_left)

                c_repeat_back = c_back.repeat(num_negative_samples, 1)
                fks_negative_back = self.fks_module_back(c_repeat_back, z_quantized_negative_back)

                fks_negative_back = fks_negative_back.view(num_negative_samples,
                                                           batch_size,
                                                           num_blocks_right
                                                           ).contiguous().permute(1, 2, 0)
                # -- compute score:
                score_matrix_back = fks_positive_back > fks_negative_back.max(2)[0]

                # -- contrastive loss
                contrastive_loss = contrastive_loss + nce_loss(fks_positive_back, fks_negative_back)
            else:
                score_matrix_back = None
                quantization_loss_negative_back = None
            ############################################################

            q_loss = quantization_loss(loss_quantization_left=quantization_loss_left,
                                       loss_quantization_negative=quantization_loss_negative,
                                       loss_quantization_right=quantization_loss_right,
                                       loss_quantization_negative_back=quantization_loss_negative_back)

            loss = contrastive_loss + self.quantization_weighting * q_loss

            # == Optim
            self.optimizer.zero_grad()
            if train:
                loss.backward()
                nn.utils.clip_grad_value_(self.parameters(), clip_value=5)
                self.optimizer.step()
            if train and (self.scheduler is not None):
                self.scheduler.step()
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
            if score_matrix_back is not None:
                accuracy = (accuracy + score_matrix_back.sum(dim=0).float() / batch_size) / 2
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
