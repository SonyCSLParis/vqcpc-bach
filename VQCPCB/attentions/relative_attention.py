import torch
import torch.nn as nn

from VQCPCB.utils import cuda_variable


class RelativeAttention(nn.Module):
    def __init__(self, head_dim, num_heads, max_seq_len):
        super(RelativeAttention, self).__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads

        self.e1 = nn.Parameter(torch.randn((num_heads * max_seq_len), self.head_dim))
        self.e2 = nn.Parameter(torch.randn((num_heads * max_seq_len), self.head_dim))

        # notes test values
        # print("!!!USING TEST VALUES!!!")
        # import numpy as np
        # aa =
        # np.arange(seq_len)
        # self.e1 = cuda_variable(torch.tensor(aa).unsqueeze(1).repeat(num_heads, head_dim).float())
        # self.e2 = cuda_variable(torch.tensor(-aa).unsqueeze(1).repeat(num_heads, head_dim).float())

    def forward(self, q):
        """

        :param q: (batch_size * num_heads, len_q, d)
        :return:
        """
        sz_b_times_n_head, len_q, d_q = q.size()
        assert sz_b_times_n_head % self.num_heads == 0
        sz_b = sz_b_times_n_head // self.num_heads

        # the trick must be done twice in case attention is full (not causal or anticausal)
        e1 = self.e1.unsqueeze(0).repeat(sz_b, 1, 1)

        # WARNING: len_q can be different from self.max_seq_len
        e1 = e1.view(sz_b * self.num_heads, self.max_seq_len, d_q)
        e1 = e1[:, :len_q]

        rel_attn_1 = torch.einsum('bld,bmd->blm',
                                  (q, e1))
        e2 = self.e2.unsqueeze(0).repeat(sz_b, 1, 1)
        # WARNING: len_q can be different from self.max_seq_len
        # TODO allow len_q > self.max_seq_len
        e2 = e2.view(sz_b * self.num_heads, self.max_seq_len, d_q)
        e2 = e2[:, :len_q]

        rel_attn_2 = torch.einsum('bld,bmd->blm',
                                  (q, e2))

        batch_size, l, _ = rel_attn_1.size()
        # ====skewing trick
        # ----Down
        # pad
        rel_attn_1 = torch.cat(
            [rel_attn_1,
             cuda_variable(torch.ones(1, 1, 1, ) * - 100).repeat(batch_size, l, 1),
             ], dim=2
        )
        rel_attn_1 = rel_attn_1.view(batch_size, l + 1, l)

        rel_attn_1 = rel_attn_1[:, :-1, :]

        # ----Up

        # pad
        # extension = cuda_variable(torch.ones(batch_size, l, 1, ) * - 100)
        rel_attn_2 = torch.cat(
            [cuda_variable(torch.ones(1, 1, 1, ) * - 100).repeat(batch_size, l, 1),
             rel_attn_2
             ], dim=2
        )
        rel_attn_2 = rel_attn_2.view(batch_size,
                                     l + 1,
                                     l,
                                     )

        rel_attn_2 = rel_attn_2[:, 1:, :]

        masks_down = torch.triu(torch.ones_like(rel_attn_1[0]).byte(),
                                diagonal=0).unsqueeze(0).repeat(sz_b_times_n_head, 1, 1).flip(
            1).flip(2).type(torch.bool)
        masks_up = torch.triu(torch.ones_like(rel_attn_2[0]).byte(),
                              diagonal=1).unsqueeze(0).repeat(sz_b_times_n_head, 1, 1).type(
            torch.bool)

        rel_attn_1 = rel_attn_1.masked_fill(masks_down, 0)
        rel_attn_2 = rel_attn_2.masked_fill(masks_up, 0)
        rel_attn = rel_attn_1 + rel_attn_2
        return rel_attn


if __name__ == '__main__':
    batch_size = 1
    head_dim = 2
    num_heads = 1
    seq_len = 6
    aa = RelativeAttention(head_dim, num_heads, seq_len)
    q = cuda_variable(torch.ones((batch_size * num_heads, seq_len, head_dim)))
    ret = aa.forward(q)
    exit()
