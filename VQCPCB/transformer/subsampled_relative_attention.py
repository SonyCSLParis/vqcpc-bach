import torch
import torch.nn as nn
from VQCPCB.utils import cuda_variable


class SubsampledRelativeAttention(nn.Module):
    def __init__(self, head_dim, num_heads, seq_len_src, seq_len_tgt):
        super(SubsampledRelativeAttention, self).__init__()
        assert seq_len_src <= seq_len_tgt
        assert seq_len_tgt % seq_len_src == 0

        self.head_dim = head_dim
        self.seq_len_src = seq_len_src
        self.seq_len_tgt = seq_len_tgt
        self.subsampling_ratio = int(seq_len_tgt // seq_len_src)
        self.num_heads = num_heads

        self.e1 = nn.Parameter(torch.randn((num_heads * seq_len_src), self.head_dim))
        self.e2 = nn.Parameter(torch.randn((num_heads * seq_len_src), self.head_dim))

        # notes test values
        # print("!!!USING TEST VALUES!!!")
        # import numpy as np
        # aa = np.arange(seq_len_src)
        # self.e1 = cuda_variable(torch.tensor(aa).unsqueeze(1).repeat(num_heads, head_dim).float())
        # self.e2 = cuda_variable(torch.tensor(-aa).unsqueeze(1).repeat(num_heads, head_dim).float())

    def forward(self, q):
        """

        :param q: (batch_size * num_heads, len_q_tgt, d)
        :return:
        """
        sz_b_times_n_head, len_q, d_q = q.size()
        assert sz_b_times_n_head % self.num_heads == 0
        sz_b = sz_b_times_n_head // self.num_heads

        batch_size = sz_b_times_n_head

        ################################
        # Causal
        e1 = self.e1.unsqueeze(0).repeat(sz_b, 1, 1)
        e1 = e1.view(sz_b * self.num_heads, self.seq_len_src, d_q)
        rel_attn_1 = torch.einsum('bld,bmd->blm', (q, e1))
        # tgt * src -> src * tgt
        rel_attn_1 = rel_attn_1.view(batch_size, self.seq_len_src, self.seq_len_tgt)

        #  one column padding on dim 2
        rel_attn_1 = torch.cat(
            [cuda_variable(torch.ones(1, 1, 1) * - 100).repeat(batch_size, self.seq_len_src, 1),
             rel_attn_1,
             ], dim=2
        )

        #  fill in with lines (ensure view can be done)
        bottom_extension = self.seq_len_tgt - self.seq_len_src
        if bottom_extension != 0:
            rel_attn_1 = torch.cat(
                [rel_attn_1,
                 cuda_variable(torch.ones(1, 1, 1) * - 100).repeat(batch_size, bottom_extension, self.seq_len_tgt + 1),
                 ], dim=1
            )

        #  skewing
        rel_attn_1 = rel_attn_1.view(batch_size, -1, self.seq_len_src)
        #  need to remove first line here
        rel_attn_1 = rel_attn_1[:, 1:]
        rel_attn_1 = rel_attn_1[:, :self.seq_len_tgt, :]
        ################################

        ################################
        #  Anticausal
        e2 = self.e2.unsqueeze(0).repeat(sz_b, 1, 1)
        e2 = e2.view(sz_b * self.num_heads, self.seq_len_src, d_q)
        rel_attn_2 = torch.einsum('bld,bmd->blm', (q, e2))

        batch_size = rel_attn_2.size(0)

        # tgt * src -> src * tgt
        rel_attn_2 = rel_attn_2.view(batch_size, self.seq_len_src, self.seq_len_tgt)

        #  one column padding on dim 2
        rel_attn_2 = torch.cat(
            [rel_attn_2,
             cuda_variable(torch.ones(1, 1, 1) * - 100).repeat(batch_size, self.seq_len_src, 1),
             ], dim=2
        )

        #  fill in with lines (ensure view can be done)
        bottom_extension = self.seq_len_tgt - self.seq_len_src
        if bottom_extension != 0:
            rel_attn_2 = torch.cat(
                [rel_attn_2,
                 cuda_variable(torch.ones(1, bottom_extension, self.seq_len_tgt + 1) * - 100).repeat(batch_size, 1, 1),
                 ], dim=1
            )

        #  SKEWWWIIIIING (tgt + 1) * (tgt + 1) -> x * tgt
        rel_attn_2 = rel_attn_2.view(batch_size, -1, self.seq_len_src)
        rel_attn_2 = rel_attn_2[:, :self.seq_len_tgt, :]
        ################################

        #  mask causal and anticausal
        masks_down = torch.triu(torch.ones(self.seq_len_src, self.seq_len_src).byte(),
                                diagonal=0).unsqueeze(0).repeat(sz_b_times_n_head, 1, 1).flip(
            1).flip(2).type(torch.bool)
        if self.subsampling_ratio != 1:
            masks_down = cuda_variable(torch.repeat_interleave(masks_down, self.subsampling_ratio, dim=1))

        masks_up = torch.triu(torch.ones(self.seq_len_src, self.seq_len_src).byte(),
                              diagonal=1).unsqueeze(0).repeat(sz_b_times_n_head, 1, 1).type(
            torch.bool)
        if self.subsampling_ratio != 1:
            masks_up = cuda_variable(torch.repeat_interleave(masks_up, self.subsampling_ratio, dim=1))

        rel_attn_1 = rel_attn_1.masked_fill(masks_up, 0)
        rel_attn_2 = rel_attn_2.masked_fill(masks_down, 0)
        rel_attn = rel_attn_1 + rel_attn_2
        return rel_attn


if __name__ == '__main__':
    batch_size = 1
    head_dim = 2
    num_heads = 1
    seq_len_src = 6
    seq_len_tgt = 6
    aa = SubsampledRelativeAttention(head_dim, num_heads, seq_len_src, seq_len_tgt)
    aa.to('cuda')
    q = cuda_variable(torch.ones((batch_size * num_heads, seq_len_tgt, head_dim)))
    ret = aa.forward(q)
    exit()
