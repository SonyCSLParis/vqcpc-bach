import torch

from VQCPCB.data_processor.data_processor import DataProcessor
from VQCPCB.utils import flatten, to_numpy


class BachDataProcessor(DataProcessor):
    def __init__(self, embedding_size, num_events, num_tokens_per_channel):
        super(BachDataProcessor, self).__init__(embedding_size=embedding_size,
                                                num_events=num_events,
                                                num_tokens_per_channel=num_tokens_per_channel
                                                )
