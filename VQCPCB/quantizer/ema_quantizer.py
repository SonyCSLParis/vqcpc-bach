from random import randint

import torch
from torch import nn

from VQCPCB.quantizer.vector_quantizer import VectorQuantizer
from VQCPCB.utils import cuda_variable


class EMAQuantizer(VectorQuantizer):
    """
    EMA as in https://arxiv.org/pdf/1711.00937.pdf
    """

    def __init__(self,
                 codebook_size,
                 codebook_dim,
                 commitment_cost,
                 initialize,
                 ema_gamma_update,
                 ema_threshold
                 ):
        super(EMAQuantizer, self).__init__()
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.embedding = nn.Parameter(torch.randn(self.codebook_size, self.codebook_dim) * 4)
        self.initialize = initialize

        self._commitment_cost = commitment_cost
        self.ema_gamma_update = ema_gamma_update
        self.ema_threshold = ema_threshold

        # Code usage monitoring (used to restart unused codes
        self.mean_usage = nn.Parameter(torch.ones(self.codebook_size) / self.codebook_size, requires_grad=False)
        self.mean_codebook = nn.Parameter(torch.randn(self.codebook_size, self.codebook_dim), requires_grad=False)

    def _initialize(self, flat_input):
        # Flatten input
        assert flat_input.size()[-1] == self.codebook_dim
        assert flat_input.size()[0] >= self.codebook_size, 'not enough elements in a batch to initialise the clusters.' \
                                                           'You need to increase the batch dimension.' \
                                                           'Just a few, 1 or 2 should be okay.'
        flat_input_rand = flat_input[torch.randperm(flat_input.size(0))]
        self.embedding.data = flat_input_rand[:self.embedding.data.size(0), :self.embedding.data.size(1)]
        self.initialize = False
        self.mean_codebook.data = self.embedding.data.detach() * self.mean_usage.data.unsqueeze(1)

    def _loss(self, inputs, quantized):
        e_latent_loss = torch.sum((quantized.detach() - inputs) ** 2, dim=-1)
        loss = self._commitment_cost * e_latent_loss
        return loss

    def forward(self, inputs, corrupt_labels=False, **kwargs):

        input_shape = inputs.size()
        z = inputs.view(-1, self.codebook_dim)
        if self.initialize:
            self._initialize(flat_input=z)
        # Calculate distances
        distances = (torch.sum(z ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding ** 2, dim=1).unsqueeze(0)
                     - 2 * torch.matmul(z, self.embedding.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # corrupt indices
        if self.training and corrupt_labels:
            random_indices = torch.randint_like(encoding_indices, low=0, high=self.codebook_size)
            mask = (torch.rand_like(random_indices.float()) > 0.05).long()
            encoding_indices_list = mask * encoding_indices + (1 - mask) * random_indices

        encoding = cuda_variable(torch.zeros(encoding_indices.shape[0], self.codebook_size))

        encoding.scatter_(dim=1, index=encoding_indices, value=1.)

        if self.training:
            self._update_ema(z, encoding)
            self._restart_codebook()
            self._ema_step()

        # Quantize and unflatten
        quantized = torch.matmul(encoding, self.embedding).view(input_shape)
        quantization_loss = self._loss(inputs, quantized)
        quantized_sg = inputs + (quantized - inputs).detach()
        encoding_indices_shape = list(input_shape[:-1])
        encoding_indices = encoding_indices.view(encoding_indices_shape)

        return quantized_sg, encoding_indices, quantization_loss

    def _restart_codebook(self):
        #  Restart by randomly choosing an other centroid if usage is below a threshold
        minimum, argmin = torch.min(self.mean_usage, dim=0)
        minimum, argmin = minimum.item(), argmin.item()
        threshold = self.ema_threshold / self.codebook_size

        batch_size = self.backup_z.size(0)
        if minimum < threshold:
            random_index = randint(0, batch_size - 1)
            self.embedding.data[argmin] = (self.backup_z[random_index].data)
            self.mean_usage[argmin].data += threshold
            self.mean_codebook.data[argmin] = (
                    self.embedding.data[argmin] * self.mean_usage[argmin].data
            )
            print(f'Restart centroid {argmin}')

    def _ema_step(self):
        self.embedding.data = self.mean_codebook / self.mean_usage.unsqueeze(1)

    def _update_ema(self, z, encoding):
        """

        :param zs: non quantized zs
        list of (batch_size, codebook_dim)
        :param encodings: list of (batch_size, codebook_size) assignations
        :return:
        """
        batch_size, z_dim = z.size()

        # compute sum_zs
        sum_z = torch.einsum('bd,bs->sd', z, encoding)
        usage = encoding.mean(0)

        # update EMAs
        self.mean_usage.data = self.mean_usage.data * self.ema_gamma_update + usage * (1 - self.ema_gamma_update)
        self.mean_codebook.data = (self.mean_codebook.data * self.ema_gamma_update
                                   + sum_z * (1 - self.ema_gamma_update) / batch_size
                                   )
        self.backup_z = z
