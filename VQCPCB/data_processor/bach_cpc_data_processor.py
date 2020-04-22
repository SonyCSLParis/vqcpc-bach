from VQCPCB.data_processor.data_processor import DataProcessor


class BachCPCDataProcessor(DataProcessor):
    def __init__(self, embedding_size, num_events, num_channels, num_tokens_per_channel, num_tokens_per_block):
        super(BachCPCDataProcessor, self).__init__(embedding_size=embedding_size,
                                                   num_events=num_events,
                                                   num_tokens_per_channel=num_tokens_per_channel
                                                   )

        self.num_tokens_per_block = num_tokens_per_block
