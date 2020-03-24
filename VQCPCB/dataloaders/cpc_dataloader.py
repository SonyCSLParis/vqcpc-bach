class CPCDataloaderGenerator:
    def __init__(self,
                 num_tokens_per_block,
                 num_blocks_left,
                 num_blocks_right,
                 negative_sampling_method,
                 num_negative_samples,
                 *args, **kwargs):

        self.num_negative_samples = num_negative_samples
        self.num_tokens_per_block = num_tokens_per_block
        self.num_blocks_left = num_blocks_left
        self.num_blocks_right = num_blocks_right
        self.negative_sampling_method = negative_sampling_method

    def dataloader(self,
                   batch_size,
                   num_workers=0
                   ):
        """

        :return: torch Dataloader, returns a dict of
        {
        'x_left': (batch_size, num_blocks_left, num_tokens_per_block)
        'x_right': (batch_size, num_blocks_right, num_tokens_per_block)
        'negative_samples': (batch_size, num_negative_samples, num_blocks_right,
        num_tokens_per_block)
        }

        """
        raise NotImplementedError

    def block_dataloader(self,
                         batch_size):
        """

            :return: torch Dataloader, returns batches of
            (batch_size, num_blocks=1, num_tokens_per_block)
            }

        """
        raise NotImplementedError
