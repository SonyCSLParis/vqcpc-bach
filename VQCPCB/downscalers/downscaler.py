from torch import nn


class Downscaler(nn.Module):
    def __init__(self, downscale_factors):
        super(Downscaler, self).__init__()
        self.downscale_factors = downscale_factors

