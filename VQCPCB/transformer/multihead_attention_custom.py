import torch
import torch.nn as nn
import torch.nn.functional as F

from VQCPCB.transformer.attentions.block_attention import BlockAttention
from VQCPCB.transformer.attentions.block_positioning import BlockPositioning
from VQCPCB.transformer.attentions.relative_attention import RelativeAttention
from VQCPCB.transformer.attentions.relative_positioning import RelativePositioning
from VQCPCB.transformer.attentions.subsampled_relative_attention import SubsampledRelativeAttention


class MultiheadAttentionCustom(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.

        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """

    def __init__(self, embed_dim, num_heads, attention_bias_type,
                 num_channels_k, num_events_k, num_channels_q, num_events_q,
                 dropout=0., bias=True, add_bias_kv=False,
                 add_zero_attn=False, kdim=None, vdim=None):
        super(MultiheadAttentionCustom, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.vdim))

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        seq_len_out = num_channels_q * num_events_q
        if attention_bias_type == 'relative_positioning':
            self.attn_bias = RelativePositioning(num_heads=num_heads,
                                                 seq_len=seq_len_out)
        elif attention_bias_type == 'relative_attention':
            self.attn_bias = RelativeAttention(head_dim=self.head_dim,
                                               num_heads=num_heads,
                                               max_seq_len=seq_len_out
                                               )
        elif attention_bias_type == 'relative_attention_target_source':
            self.attn_bias = SubsampledRelativeAttention(
                head_dim=self.head_dim,
                num_heads=num_heads,
                seq_len_src=num_channels_k * num_events_k,
                seq_len_tgt=num_channels_q * num_events_q
            )
        elif attention_bias_type == 'block_positioning':
            self.attn_bias = BlockPositioning(head_dim=1,  # has to be 1 in that case
                                              num_heads=num_heads,
                                              num_channels_k=num_channels_k,
                                              num_events_k=num_events_k,
                                              num_channels_q=num_channels_q,
                                              num_events_q=num_events_q
                                              )
        elif attention_bias_type == 'block_attention':
            self.attn_bias = BlockAttention(head_dim=self.head_dim,
                                            num_heads=num_heads,
                                            num_channels_k=num_channels_k,
                                            num_events_k=num_events_k,
                                            num_channels_q=num_channels_q,
                                            num_events_q=num_events_q
                                            )
        elif attention_bias_type == 'no_bias':
            self.attn_bias = None
        else:
            raise ValueError(f'{attention_bias_type} is not a valid type of attention bias')

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None,
                static_k=None, static_v=None):
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).

    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.

        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """

        qkv_same = torch.equal(query, key) and torch.equal(key, value)
        kv_same = torch.equal(key, value)

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5

        ################################################
        # get q, k and v
        if self._qkv_same_embed_dim:
            if qkv_same:
                # self-attention
                q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)

            elif kv_same:
                # encoder-decoder attention
                # This is inline in_proj function with self.in_proj_weight and self.in_proj_bias
                _b = self.in_proj_bias
                _start = 0
                _end = embed_dim
                _w = self.in_proj_weight[_start:_end, :]
                if _b is not None:
                    _b = _b[_start:_end]
                q = F.linear(query, _w, _b)

                if key is None:
                    assert value is None
                    k = None
                    v = None
                else:

                    # This is inline in_proj function with self.in_proj_weight and self.in_proj_bias
                    _b = self.in_proj_bias
                    _start = embed_dim
                    _end = None
                    _w = self.in_proj_weight[_start:, :]
                    if _b is not None:
                        _b = _b[_start:]
                    k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

            else:
                # This is inline in_proj function with self.in_proj_weight and self.in_proj_bias
                _b = self.in_proj_bias
                _start = 0
                _end = embed_dim
                _w = self.in_proj_weight[_start:_end, :]
                if _b is not None:
                    _b = _b[_start:_end]
                q = F.linear(query, _w, _b)

                # This is inline in_proj function with self.in_proj_weight and self.in_proj_bias
                _b = self.in_proj_bias
                _start = embed_dim
                _end = embed_dim * 2
                _w = self.in_proj_weight[_start:_end, :]
                if _b is not None:
                    _b = _b[_start:_end]
                k = F.linear(key, _w, _b)

                # This is inline in_proj function with self.in_proj_weight and self.in_proj_bias
                _b = self.in_proj_bias
                _start = embed_dim * 2
                _end = None
                _w = self.in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                v = F.linear(value, _w, _b)
        else:
            q_proj_weight_non_opt = torch.jit._unwrap_optional(self.q_proj_weight)
            len1, len2 = q_proj_weight_non_opt.size()
            assert len1 == embed_dim and len2 == query.size(-1)

            k_proj_weight_non_opt = torch.jit._unwrap_optional(self.k_proj_weight)
            len1, len2 = k_proj_weight_non_opt.size()
            assert len1 == embed_dim and len2 == key.size(-1)

            v_proj_weight_non_opt = torch.jit._unwrap_optional(self.v_proj_weight)
            len1, len2 = v_proj_weight_non_opt.size()
            assert len1 == embed_dim and len2 == value.size(-1)

            if self.in_proj_bias is not None:
                q = F.linear(query, q_proj_weight_non_opt, self.in_proj_bias[0:embed_dim])
                k = F.linear(key, k_proj_weight_non_opt, self.in_proj_bias[embed_dim:(embed_dim * 2)])
                v = F.linear(value, v_proj_weight_non_opt, self.in_proj_bias[(embed_dim * 2):])
            else:
                q = F.linear(query, q_proj_weight_non_opt, self.in_proj_bias)
                k = F.linear(key, k_proj_weight_non_opt, self.in_proj_bias)
                v = F.linear(value, v_proj_weight_non_opt, self.in_proj_bias)
        q = q * scaling

        ################################################
        # biases in k or v
        if self.bias_k is not None and self.bias_v is not None:
            if static_k is None and static_v is None:
                k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
                v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
                if attn_mask is not None:
                    attn_mask = torch.cat([attn_mask,
                                           torch.zeros((attn_mask.size(0), 1),
                                                       dtype=attn_mask.dtype,
                                                       device=attn_mask.device)], dim=1)
                if key_padding_mask is not None:
                    key_padding_mask = torch.cat(
                        [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                                       dtype=key_padding_mask.dtype,
                                                       device=key_padding_mask.device)], dim=1)
            else:
                assert static_k is None, "bias cannot be added to static key."
                assert static_v is None, "bias cannot be added to static value."
        else:
            assert self.bias_k is None
            assert self.bias_v is None

        ################################################
        # flatten batch and head dims
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        ################################################
        # static k and v (if you want to impose a certain k and v)
        if static_k is not None:
            assert static_k.size(0) == bsz * self.num_heads
            assert static_k.size(2) == head_dim
            k = static_k

        if static_v is not None:
            assert static_v.size(0) == bsz * self.num_heads
            assert static_v.size(2) == head_dim
            v = static_v

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, torch.zeros((attn_mask.size(0), 1),
                                                              dtype=attn_mask.dtype,
                                                              device=attn_mask.device)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                                   dtype=key_padding_mask.dtype,
                                                   device=key_padding_mask.device)], dim=1)

        ################################################
        # compute Q.K
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)

        ################################################
        #  compute attention biases (relative or block attentions)
        if self.attn_bias is not None:
            attn_bias = self.attn_bias(q)
            attn_output_weights += attn_bias


        ################################################
        # compute A
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        ################################################
        #  compute V
        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights
        else:
            return attn_output, None


if __name__ == '__main__':
    # input seq
    batch_size = 10
    num_channels = 3
    num_events = 4
    seq_len = num_channels * num_events

    # heads
    embed_dim = 64
    num_heads = 4
    attention_bias_type = 'block_attention'
    attention_mask_shape = 'full'

    multi_head_attention = MultiheadAttentionCustom(
        embed_dim=embed_dim,
        num_heads=num_heads,
        attention_bias_type=attention_bias_type,
        attention_mask_shape=attention_mask_shape,
        num_channels_k=num_channels,
        num_events_k=num_events,
        num_channels_q=num_channels,
        num_events_q=num_events,
    )

    x = torch.arange(0, seq_len)\
        .unsqueeze(1).unsqueeze(2)\
        .repeat(1, batch_size, embed_dim)\
        .float()

    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    multi_head_attention(query=x,
                         key=x,
                         value=x,
                         attn_mask=mask)
