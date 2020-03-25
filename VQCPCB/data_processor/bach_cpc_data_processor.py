import numpy as np
import torch

from VQCPCB.data_processor.data_processor import DataProcessor
from VQCPCB.utils import to_numpy


class BachCPCDataProcessor(DataProcessor):
    def __init__(self, embedding_size, num_events, num_channels, num_tokens_per_channel, num_tokens_per_block):
        super(BachCPCDataProcessor, self).__init__(embedding_size=embedding_size,
                                                   num_events=num_events,
                                                   num_tokens_per_channel=num_tokens_per_channel
                                                   )

        self.num_tokens_per_block = num_tokens_per_block

    def preprocess(self, x, device):
        """
        Preprocess a dcpc block

        :param x: (..., num_ticks, num_voices) of appropriate dimensions
        :return: (..., num_blocks, num_tokens_per_block)
        """
        # if flat_input:

        num_ticks, num_voices = x.size()[-2:]
        remaining_dims = x.size()[:-2]

        x = x.view(-1, num_ticks, num_voices).contiguous()
        x = x.view(-1, num_voices * num_ticks)

        assert x.size(1) % self.num_tokens_per_block == 0
        x = x.split(self.num_tokens_per_block, dim=1)
        x = torch.cat(
            [t.unsqueeze(1) for t in x], dim=1
        )

        num_blocks = x.size(1)
        x = x.view(*remaining_dims, num_blocks, self.num_tokens_per_block)
        return x.long().to(device)

    def embed(self, block):
        """
        Embed a DCPC block

        :param block: (..., num_tokens_per_block)
        :return: (..., num_tokens_per_block, embedding_size)
        """
        batch_dims = block.size()[:-1]
        product_batch_dim = np.prod(batch_dims)
        num_tokens_per_block = block.size()[-1]
        block = block.view(-1, num_tokens_per_block)

        block = block.view(product_batch_dim, -1, self.num_channels).long()

        separate_voices = block.split(split_size=1, dim=2)

        separate_voices = [
            embedding(voice[:, :, 0])[:, :, None, :]
            for voice, embedding
            in zip(separate_voices, self.embeddings)
        ]
        x = torch.cat(separate_voices, 2)
        # TODO check with measures

        x = x.view(product_batch_dim, -1, self.embedding_size)
        x = x.view(*batch_dims, num_tokens_per_block, self.embedding_size)
        return x

    def postprocess(self, list_elements):
        tensor_score = torch.cat(list_elements, dim=0)
        tensor_score = to_numpy(tensor_score)
        return tensor_score

    def write(self, score, path):
        score.write('xml', f'{path}.xml')
        return