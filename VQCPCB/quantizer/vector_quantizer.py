import torch
from torch import nn

from VQCPCB.utils import cuda_variable


class VectorQuantizer(nn.Module):
    def __init__(self, **kwargs):
        super(VectorQuantizer, self).__init__()

    def forward(self, inputs, **kwargs):
        raise NotImplementedError


class NoQuantization(VectorQuantizer):
    def __init__(self, codebook_dim):
        super(NoQuantization, self).__init__()
        self.codebook_dim = codebook_dim

    def forward(self, inputs, **kwargs):
        loss = cuda_variable(torch.zeros_like(inputs)).sum(dim=-1)
        quantized_sg = inputs
        encoding_indices = None
        return quantized_sg, encoding_indices, loss


class ProductVectorQuantizer(VectorQuantizer):
    def __init__(self,
                 codebook_size,
                 codebook_dim,
                 commitment_cost,
                 use_batch_norm,
                 initialize,
                 squared_l2_norm
                 ):
        super(ProductVectorQuantizer, self).__init__()
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self._commitment_cost = commitment_cost

        self.embedding = nn.Parameter(torch.randn(self.codebook_size, self.codebook_dim) * 4)
        self.initialize = initialize
        self.squared_l2_norm = squared_l2_norm
        self.use_batch_norm = use_batch_norm

        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(codebook_dim)

    def _initialize(self, flat_input):
        # Flatten input
        assert flat_input.size()[-1] == self.codebook_dim
        assert flat_input.size()[0] >= self.codebook_size, 'not enough elements in a batch to initialise the clusters.' \
                                                           'You need to increase the batch dimension.' \
                                                           'Just a few, 1 or 2 should be okay.'
        flat_input_rand = flat_input[torch.randperm(flat_input.size(0))]
        self.embedding.data = flat_input_rand[:self.embedding.data.size(0), :self.embedding.data.size(1)]
        self.initialize = False

    def _loss(self, inputs, quantized):
        if self.squared_l2_norm:
            e_latent_loss = torch.sum((quantized.detach() - inputs) ** 2, dim=-1)
            q_latent_loss = torch.sum((quantized - inputs.detach()) ** 2, dim=-1)
        else:
            epsilon = 1e-5
            e_latent_loss = torch.norm((quantized.detach() - inputs) + epsilon,
                                       dim=-1)
            q_latent_loss = torch.norm((quantized - inputs.detach()) + epsilon,
                                       dim=-1)
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        return loss

    def forward(self, inputs, corrupt_labels=False, **kwargs):

        input_shape = inputs.size()

        # Normalize and flatten
        if self.use_batch_norm:

            flat_input = inputs.view(-1, self.codebook_dim).unsqueeze(1)

            flat_input = flat_input.permute(0, 2, 1)
            flat_input = self.batch_norm(flat_input)
            flat_input = flat_input.permute(0, 2, 1).contiguous()
            flat_input = flat_input[:, 0, :]
        else:
            flat_input = inputs.view(-1, self.codebook_dim)

        if self.initialize:
            self._initialize(flat_input=flat_input)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding ** 2, dim=1).unsqueeze(0)
                     - 2 * torch.matmul(flat_input, self.embedding.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # corrupt indices
        if self.training and corrupt_labels:
            random_indices = torch.randint_like(encoding_indices, low=0, high=self.codebook_size)
            mask = (torch.rand_like(random_indices.float()) > 0.05).long()
            encoding_indices_list = mask * encoding_indices + (1 - mask) * random_indices

        encoding = cuda_variable(torch.zeros(encoding_indices.shape[0], self.codebook_size))

        encoding.scatter_(dim=1, index=encoding_indices, value=1.)

        # Quantize and unflatten
        quantized = torch.matmul(encoding, self.embedding).view(input_shape)

        quantization_loss = self._loss(inputs, quantized)

        quantized_sg = inputs + (quantized - inputs).detach()

        encoding_indices_shape = list(input_shape[:-1])
        encoding_indices = encoding_indices.view(encoding_indices_shape)

        return quantized_sg, encoding_indices, quantization_loss
