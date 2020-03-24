import numpy as np
import torch

from VQCPCB.attentions.block_positioning import BlockPositioning


class BlockAttention(BlockPositioning):
    """

    Block Self-attention.
    Blocks per time and per voice
    """

    def __init__(self, head_dim, num_heads,
                 num_channels_k, num_events_k, num_channels_q, num_events_q):
        super(BlockAttention, self).__init__(head_dim, num_heads,
                                             num_channels_k, num_events_k, num_channels_q, num_events_q)
        return

    def forward(self, q):
        """

        :param q: (batch_size * num_heads, len_q, d)
        :return:
        """
        q_channel = q
        q_event = q
        sz_b = int(q.size()[0] / self.num_heads)

        event_blocks, channel_blocks = self.prepare_blocks()

        def repeat_batch_dim(r_matrix):
            r_matrix = r_matrix.permute(3, 0, 1, 2)
            # batch is moving slower in q, so repeat like this is okay here
            r_matrix = r_matrix.repeat(sz_b, 1, 1, 1)
            return r_matrix

        channel_blocks = repeat_batch_dim(channel_blocks)
        event_blocks = repeat_batch_dim(event_blocks)

        rel_attn_channels = torch.einsum('bld,blmd->blm', (q_channel, channel_blocks))
        rel_attn_events = torch.einsum('bld,blmd->blm', (q_event, event_blocks))
        rel_attn = rel_attn_channels + rel_attn_events
        return rel_attn


if __name__ == '__main__':
    batch_size = 10
    num_channels_in = 3
    num_events_in = 4
    num_channels_out = 3
    num_events_out = 4
    head_dim = 4
    num_heads = 8

    aa = BlockAttention(head_dim, num_heads,
                        num_channels_k=num_channels_in, num_events_k=num_events_in,
                        num_channels_q=num_channels_out, num_events_q=num_events_out)

    q = torch.arange(0, num_channels_in * num_events_in)\
        .unsqueeze(0).unsqueeze(2)\
        .repeat(batch_size * num_heads, 1, head_dim)\
        .float()

    ret = aa.forward(q)
