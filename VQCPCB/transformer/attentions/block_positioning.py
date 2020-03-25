import torch
import torch.nn as nn

from VQCPCB.utils import cuda_variable


class BlockPositioning(nn.Module):
    def __init__(self, head_dim, num_heads,
                 num_channels_k, num_events_k, num_channels_q, num_events_q):
        super(BlockPositioning, self).__init__()
        self.head_dim = head_dim
        self.num_channels_k = num_channels_k
        self.num_events_k = num_events_k
        self.num_channels_q = num_channels_q
        self.num_events_q = num_events_q
        self.seq_len_k = self.num_channels_k * self.num_events_k
        self.seq_len_q = self.num_channels_q * self.num_events_q
        self.num_events_max = max(num_events_q, num_events_k)
        self.num_heads = num_heads

        self.channel_blocks = nn.Parameter(torch.randn((num_channels_k, num_channels_q, head_dim, num_heads)))
        self.event_blocks = nn.Parameter(torch.randn((self.num_events_max, head_dim, num_heads)))
        self.event_blocks_future = nn.Parameter(torch.randn((self.num_events_max, head_dim, num_heads)))

        # notes: Test values
        # print("!!!USING TEST VALUES!!!")
        # aa = np.arange(self.num_channels_k * self.num_channels_q).reshape(self.num_channels_k, self.num_channels_q)
        # self.channel_blocks = torch.tensor(aa) \
        #     .unsqueeze(2).unsqueeze(3).repeat(1, 1, head_dim, num_heads)
        # self.event_blocks = torch.tensor(np.arange(self.num_events_max)) \
        #     .unsqueeze(1).unsqueeze(2).repeat(1, head_dim, num_heads)
        # self.event_blocks_future = torch.tensor(-np.arange(self.num_events_max)) \
        #     .unsqueeze(1).unsqueeze(2).repeat(1, head_dim, num_heads)

    def forward(self, q):
        """

        :param q: (batch_size * num_heads, len_q, d)
        :return:
        """
        batch_x_heads = q.shape[0]
        batch_dim = batch_x_heads // self.num_heads

        event_blocks, channel_blocks = self.prepare_blocks()

        ret = event_blocks + channel_blocks
        ret = ret.squeeze()
        ret = ret.permute(2, 0, 1).repeat(batch_dim, 1, 1)
        return ret

    def prepare_blocks(self):
        # channel blocks
        channel_blocks = cuda_variable(self.channel_blocks.repeat(self.num_events_k, self.num_events_q, 1, 1).float())

        # event blocks
        event_blocks = cuda_variable(
            torch.zeros(self.seq_len_k, self.seq_len_q, self.head_dim, self.num_heads).float()) - 111

        # fill in the causal part
        for delta_t in range(self.num_events_max):
            for event_in in range(self.num_events_k):
                event_out = event_in - delta_t
                if event_out < 0:
                    continue
                event_in_start = event_in * self.num_channels_k
                event_in_end = (event_in + 1) * self.num_channels_k
                event_out_start = event_out * self.num_channels_q
                event_out_end = (event_out + 1) * self.num_channels_q
                event_blocks[event_in_start:event_in_end, event_out_start:event_out_end] = self.event_blocks[delta_t]


        # fill in the anticausal part
        for delta_t in range(self.num_events_max):
            for event_in in range(self.num_events_k):
                event_out = event_in + delta_t
                if event_out >= self.num_events_q:
                    continue
                event_in_start = event_in * self.num_channels_k
                event_in_end = (event_in + 1) * self.num_channels_k
                event_out_start = event_out * self.num_channels_q
                event_out_end = (event_out + 1) * self.num_channels_q
                event_blocks[event_in_start:event_in_end, event_out_start:event_out_end] = self.event_blocks_future[
                    delta_t]
        return event_blocks, channel_blocks


if __name__ == '__main__':
    batch_size = 10
    num_channels_k = 3
    num_events_k = 4
    num_channels_q= 3
    num_events_q = 4
    head_dim = 4
    num_heads = 8

    aa = BlockPositioning(head_dim, num_heads,
                          num_channels_k=num_channels_k, num_events_k=num_events_k,
                          num_channels_q=num_channels_q, num_events_q=num_events_q)
    q = torch.ones((batch_size * num_heads, num_channels_k*num_events_k, head_dim))
    ret = aa.forward(q)
