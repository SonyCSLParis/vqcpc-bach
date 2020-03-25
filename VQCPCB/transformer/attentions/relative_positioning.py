import numpy as np
import torch
import torch.nn as nn

from VQCPCB.utils import cuda_variable


class RelativePositioning(nn.Module):
    def __init__(self, num_heads, seq_len):
        super(RelativePositioning, self).__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads

        self.e1 = nn.Parameter(torch.randn(seq_len, num_heads))
        self.e2 = nn.Parameter(torch.randn(seq_len - 1, num_heads))

        # notes: test values
        # print("!!!USING TEST VALUES!!!")
        # aa = np.arange(self.seq_len)
        # self.e1 = torch.tensor(aa).unsqueeze(1).repeat(1, num_heads).float()
        # bb = -np.arange(self.seq_len - 1) - 1
        # self.e2 = torch.tensor(bb).unsqueeze(1).repeat(1, num_heads).float()

    def forward(self, q):
        """

        :param q: (batch_size * num_heads, len_q, d)
        :return:
        """

        batch_x_heads = q.shape[0]
        batch_dim = batch_x_heads // self.num_heads

        ret = cuda_variable(torch.zeros(self.seq_len, self.seq_len, self.num_heads)) - 111

        #  positive tau
        for delta_t in range(self.seq_len):
            diag_indices = np.tril_indices(self.seq_len, -delta_t)
            ret[diag_indices] = self.e1[delta_t]
        # negative tau
        for delta_t in range(self.seq_len - 1):
            diag_indices = np.triu_indices(self.seq_len, delta_t + 1)
            ret[diag_indices] = self.e2[delta_t]

        ret = ret.permute(2, 0, 1).repeat(batch_dim, 1, 1)

        return ret


if __name__ == '__main__':
    batch_size = 5
    seq_len = 3
    head_dim = 2
    num_heads = 8
    aa = RelativePositioning(num_heads, seq_len, 'full')
    q = torch.ones((batch_size * num_heads, seq_len, head_dim))
    ret = aa.forward(q)
