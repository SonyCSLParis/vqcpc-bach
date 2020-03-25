import torch
import torch.nn as nn


def nce_loss(positive, negatives):
    """

    :param positive: b * k
    :param negatives: b * k * num_negatives
    :return:
    """

    positive = positive
    negatives = negatives

    negatives_and_positive = torch.cat(
        [
            negatives,
            positive.unsqueeze(2)
        ], dim=2
    )
    normalizer = torch.logsumexp(negatives_and_positive, dim=2)

    loss_batch = positive - normalizer

    # sum over k, mean over batches
    loss = -loss_batch.sum(1).mean(0)
    # loss = -torch.mean(loss_batch)
    return loss


def quantization_loss(loss_quantization_left,
                      loss_quantization_negative,
                      loss_quantization_right):
    loss_quantization = torch.cat(
        (loss_quantization_left.sum(1),
         loss_quantization_right.sum(1),
         loss_quantization_negative.sum(3).sum(2).sum(1),
         ), dim=0
    ).mean()
    return loss_quantization


class CModule(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim,
                 num_layers, dropout):
        super(CModule, self).__init__()

        self.g_ar_fwd = torch.nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )

        self.output_linear = nn.Linear(hidden_size, output_dim)

    def forward(self, zs, h):
        c, h = self.g_ar_fwd(zs, h)
        # take last time step
        c = c[:, -1]
        c = self.output_linear(c)
        return c


class FksModule(nn.Module):
    def __init__(self, z_dim, c_dim, k_max):
        super(FksModule, self).__init__()
        # f_k
        self.k_max = k_max
        self.W = nn.Parameter(torch.randn(z_dim, c_dim, k_max))

    def forward(self, c_t, zs):
        """

        :param c_t:
        :param zs:
        :return: log of fks
        """
        batch_size = c_t.shape[0]
        W_c = torch.matmul(c_t, self.W).permute(1, 2, 0)
        product = torch.matmul(W_c.view(batch_size, self.k_max, 1, -1),
                               zs.view(batch_size, self.k_max, -1, 1))
        fks = torch.squeeze(product)
        return fks
