import os
from itertools import islice

import numpy as np
import torch
from tqdm import tqdm

from VQCPCB.encoder import EncoderTrainer
from VQCPCB.utils import categorical_crossentropy, flatten, unflatten, cuda_variable, distilled_categorical_crossentropy


class StudentEncoderTrainer(EncoderTrainer):
    def __init__(self,
                 model_dir,
                 dataloader_generator,
                 encoder,
                 num_events_masked,
                 teacher,
                 auxiliary_decoder,
                 quantization_weighting,
                 num_gpus=1,  # todo add Dataparallel
                 ):
        super(StudentEncoderTrainer, self).__init__(
            dataloader_generator=dataloader_generator
        )
        self.model_dir = model_dir

        self.dataloader_generator = dataloader_generator
        self.encoder = encoder
        # auxiliary networks
        self.teacher = teacher
        self.auxiliary_decoder = auxiliary_decoder

        # compute information from encoder network
        self.codebook_dim = self.encoder.quantizer.codebook_dim
        self.upscale_factors = self.auxiliary_decoder.upscale_factors

        self.num_tokens_per_channel = self.encoder.data_processor.num_tokens_per_channel
        self.num_channels = len(self.num_tokens_per_channel)
        self.num_events_masked = num_events_masked

        assert self.encoder.data_processor.num_tokens % np.prod(self.upscale_factors) == 0

        self.quantization_weighting = quantization_weighting

        # optim
        self.optimizer = None
        self.scheduler = None

    def init_optimizers(self, lr=1e-3):
        self.optimizer_enc_dec = torch.optim.Adam(
            list(self.auxiliary_decoder.parameters()) +
            list(self.encoder.parameters()),
            lr=lr
        )

        self.optimizer_teacher = torch.optim.Adam(
            self.teacher.parameters(),
            lr=lr
        )

    def to(self, device):
        self.auxiliary_decoder.to(device)
        self.encoder.to(device)
        self.teacher.to(device)

    def save(self, early_stopped):
        if early_stopped:
            model_dir = f'{self.model_dir}/early_stopped'
        else:
            model_dir = f'{self.model_dir}/overfitted'

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.encoder.save(early_stopped=early_stopped)
        torch.save(self.auxiliary_decoder.state_dict(), f'{model_dir}/decoder')
        torch.save(self.teacher.state_dict(), f'{model_dir}/teacher')
        # print(f'Model {model_dir} saved')

    def load(self, early_stopped):
        print(f'Loading models {self.__repr__()}')
        print(f'Loading from {self.model_dir}')
        if early_stopped:
            model_dir = f'{self.model_dir}/early_stopped'
        else:
            model_dir = f'{self.model_dir}/overfitted'
        self.encoder.load(early_stopped=early_stopped)
        self.auxiliary_decoder.load_state_dict(torch.load(f'{model_dir}/decoder'))
        self.teacher.load_state_dict(torch.load(f'{model_dir}/teacher'))

    def train(self):
        self.encoder.train()
        self.auxiliary_decoder.train()
        self.teacher.train()

    def eval(self):
        self.encoder.eval()
        self.auxiliary_decoder.eval()
        self.teacher.eval()

    def forward_teacher(self, x):
        # batch_size, num_events, num_channels = x.size()
        target = x
        masked_x, notes_to_be_predicted = self.mask_teacher(
            x,
            num_events_masked=self.num_events_masked
        )

        masked_x_embedded = self.teacher.data_processor.embed(masked_x)
        # NOTE teacher uses the same embedding as the encoder
        weights_per_category = self.teacher(masked_x_embedded)

        loss = categorical_crossentropy(value=weights_per_category,
                                        target=target,
                                        mask=notes_to_be_predicted)
        loss = loss.mean()
        return {'loss':                  loss,
                'notes_to_be_predicted': notes_to_be_predicted,
                'weights_per_category':  weights_per_category,
                'monitored_quantities':  {
                    'loss_teacher': loss.item()
                }
                }

    def mask_teacher(self, x, num_events_masked):
        """

        :param x: (batch_size, num_events, num_channels)
        :param num_events_masked: number of events to be masked (before and after) the
        masked_event_index
        :return:
        """
        input = flatten(x)
        batch_size, sequence_length = input.size()
        num_events = sequence_length // self.num_channels
        assert sequence_length % self.num_channels == 0

        # TODO different masks for different elements in the batch
        # leave num_events_masked events before and num_events_masked after
        masked_event_index = torch.randint(high=num_events,
                                           size=()).item()

        # the mask indices are precisely the self.num_notes_per_voice
        notes_to_be_predicted = torch.zeros_like(input)

        notes_to_be_predicted[:,
        masked_event_index * self.num_channels
        :(masked_event_index + 1) * self.num_channels] = 1

        mask_tokens = cuda_variable(torch.LongTensor(self.num_tokens_per_channel))
        mask_tokens = mask_tokens.unsqueeze(0).repeat(batch_size, num_events)

        notes_to_mask = torch.zeros_like(input)
        notes_to_mask[:,
        max((masked_event_index - num_events_masked) * self.num_channels, 0)
        :(masked_event_index + num_events_masked + 1) * self.num_channels] = 1

        masked_input = input * (1 - notes_to_mask) + mask_tokens * notes_to_mask

        # unflatten
        masked_x = unflatten(masked_input,
                             self.num_channels)
        notes_to_be_predicted = unflatten(notes_to_be_predicted,
                                          self.num_channels)
        return masked_x, notes_to_be_predicted

    def forward_encdec(self, x,
                       weights_per_category_teacher,
                       notes_to_be_predicted):
        """

        :param x: (batch_size, num_events, num_channels)
        :param weights_per_category_teacher: list of (batch_size, chorale_length, num_tokens_of_corresponding_channel)
        :param notes_to_be_predicted:
        :return:
        """
        # detach weights_per_category_teacher
        weights_per_category_teacher = [t.detach()
                                        for t in weights_per_category_teacher]

        z_quantized, encoding_indices, quantization_loss = self.encoder(x)
        weights_per_category = self.auxiliary_decoder(z_quantized)

        reconstruct_loss = distilled_categorical_crossentropy(value=weights_per_category,
                                                              target=weights_per_category_teacher,
                                                              mask=notes_to_be_predicted)

        # quantization loss is (batch_size, downsampled_sequence_size)
        # mean over sequence_size or sum ?
        loss = self.quantization_weighting * quantization_loss.mean() + reconstruct_loss.mean()

        return {'loss':                 loss,
                'monitored_quantities': {
                    'loss_quantization':   quantization_loss.mean().item(),
                    'loss_reconstruction': reconstruct_loss.mean().item(),
                    'loss_encdec':         loss.item(),
                    'loss_monitor':        reconstruct_loss.mean().item(),
                }
                }

    def epoch(self, data_loader,
              train=True,
              num_batches=None,
              corrupt_labels=False,
              ):

        means = None

        if train:
            self.train()
        else:
            self.eval()

        for sample_id, tensor_dict in tqdm(enumerate(
                islice(data_loader, num_batches)),
                ncols=80):

            # ========Train teacher ==================
            self.optimizer_teacher.zero_grad()

            x = tensor_dict['x']
            x = self.teacher.data_processor.preprocess(x)

            forward_pass_teacher = self.forward_teacher(x)

            loss_teacher = forward_pass_teacher['loss']
            notes_to_be_predicted = forward_pass_teacher['notes_to_be_predicted']
            weights_per_category_teacher = forward_pass_teacher['weights_per_category']

            if train:
                loss_teacher.backward()
                torch.nn.utils.clip_grad_norm_(self.teacher.parameters(), 5)
                self.optimizer_teacher.step()

            # ========Train encoder-decoder =============
            self.optimizer_enc_dec.zero_grad()
            weights_per_category_teacher = [w.detach()
                                            for w in weights_per_category_teacher]
            forward_pass_encdec = self.forward_encdec(x,
                                                      weights_per_category_teacher,
                                                      notes_to_be_predicted)

            loss_encdec = forward_pass_encdec['loss']
            if train:
                loss_encdec.backward()
                torch.nn.utils.clip_grad_norm_(self.auxiliary_decoder.parameters(), 5)
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 5)
                self.optimizer_enc_dec.step()

            # Monitored quantities
            monitored_quantities = dict(forward_pass_teacher['monitored_quantities'],
                                        **forward_pass_encdec['monitored_quantities'])

            # average quantities
            if means is None:
                means = {key: 0
                         for key in monitored_quantities}
            means = {
                key: value + means[key]
                for key, value in monitored_quantities.items()
            }

            del loss_teacher

        # renormalize monitored quantities
        means = {
            key: value / (sample_id + 1)
            for key, value in means.items()
        }
        return means