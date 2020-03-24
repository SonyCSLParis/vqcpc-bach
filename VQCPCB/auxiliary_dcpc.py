import torch
import torch.nn as nn

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

    # TODO to remove just a test!!!!
    # def forward(self, zs, h):
    #     return zs.mean(1)
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
        # TODO rescale?!
        # W_c = W_c / torch.sqrt(self.W.size(0))

        product = torch.matmul(W_c.view(batch_size, self.k_max, 1, -1),
                               zs.view(batch_size, self.k_max, -1, 1))
        fks = torch.squeeze(product)
        return fks