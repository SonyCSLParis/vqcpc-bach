import torch
from torch import nn


class VectorQuantizer(nn.Module):
    def __init__(self, **kwargs):
        super(VectorQuantizer, self).__init__()

    def forward(self, inputs, **kwargs):
        raise NotImplementedError


class NoQuantization(VectorQuantizer):
    def forward(self, inputs, **kwargs):
        loss = torch.zeros_like(inputs).cuda().sum(dim=-1)
        quantized_sg = inputs
        encoding_indices = None
        return quantized_sg, encoding_indices, loss


class ProductVectorQuantizer(VectorQuantizer):
    def __init__(self,
                 codebook_size,
                 codebook_dim,
                 commitment_cost,
                 num_codebooks,
                 use_batch_norm,
                 initialize,
                 squared_l2_norm
                 ):
        super(ProductVectorQuantizer, self).__init__()
        self.num_codebooks = num_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self._commitment_cost = commitment_cost

        assert self.codebook_dim % self.num_codebooks == 0
        self.embeddings = nn.ParameterList([nn.Parameter(
            torch.randn(self.codebook_size, self.codebook_dim // num_codebooks) * 4)
            for _ in range(num_codebooks)
        ]
        )

        self.initialize = initialize
        self.squared_l2_norm = squared_l2_norm
        self.use_batch_norm = use_batch_norm

        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(codebook_dim)

    def _initialize(self, flat_input):
        # Flatten input
        assert flat_input.size()[-1] == self.codebook_dim

        for k, embedding in enumerate(self.embeddings):
            flat_input_rand = flat_input[torch.randperm(flat_input.size(0))]
            embedding.data = flat_input_rand[
                             :embedding.data.size(0),
                             k * embedding.data.size(1): (k + 1) * embedding.data.size(1)]

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
        distances = [(torch.sum(input_component ** 2, dim=1, keepdim=True)
                      + torch.sum(embedding ** 2, dim=1)
                      - 2 * torch.matmul(input_component, embedding.t()))
                     for input_component, embedding
                     in zip(
                flat_input.chunk(chunks=self.num_codebooks, dim=1),
                self.embeddings)]

        # Encoding
        encoding_indices_list = [torch.argmin(distance, dim=1).unsqueeze(1)
                                 for distance in distances]

        # corrupt indices
        if self.training and corrupt_labels:
            random_indices_list = [torch.randint_like(encoding_indices_list[0],
                                                      low=0, high=self.codebook_size)
                                   for _ in range(self.num_codebooks)
                                   ]
            mask_list = [(torch.rand_like(random_indices.float()) > 0.05).long()
                         for random_indices in random_indices_list]
            encoding_indices_list = [mask * encoding_indices + (1 - mask) * random_indices
                                     for encoding_indices, random_indices, mask
                                     in zip(
                    encoding_indices_list,
                    random_indices_list,
                    mask_list)
                                     ]
        # FIX the code is not non-cuda compatible
        # encodings = [torch.zeros(encoding_indices.shape[0], self.codebook_size)
        encodings = [torch.zeros(encoding_indices.shape[0], self.codebook_size).cuda(
            non_blocking=True)
            for encoding_indices in encoding_indices_list]
        for encoding, encoding_indices in zip(encodings, encoding_indices_list):
            encoding.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized_list = [torch.matmul(encoding, embedding)
                          for encoding, embedding
                          in zip(encodings, self.embeddings)
                          ]
        quantized = torch.cat(quantized_list, dim=1).view(input_shape)

        quantization_loss = self._loss(inputs, quantized)

        quantized_sg = inputs + (quantized - inputs).detach()

        # encoding_indices = torch.zeros_like(encoding_indices_list[0])
        # for encoding_index in encoding_indices_list:
        #     encoding_indices = encoding_indices * self.codebook_size + encoding_index
        # print(len(torch.unique(encoding_indices)))
        # encoding_indices = encoding_indices.view(input_shape[:-1])

        encoding_indices_shape = list(input_shape[:-1]) + [-1]
        encoding_indices = torch.stack(encoding_indices_list, dim=-1).view(encoding_indices_shape)

        return quantized_sg, encoding_indices, quantization_loss