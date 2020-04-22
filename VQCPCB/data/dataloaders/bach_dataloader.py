import music21
from VQCPCB.data.chorale_dataset import ChoraleBeatsDataset

subdivision = 4
num_voices = 4


class BachDataloaderGenerator:
    def __init__(self, sequences_size):
        super(BachDataloaderGenerator, self).__init__()
        dataset = ChoraleBeatsDataset(
            corpus_it_gen=music21.corpus.chorales.Iterator,
            voice_ids=[0, 1, 2, 3],
            metadatas=[],
            sequences_size=sequences_size,
            subdivision=subdivision,
        )
        self.dataset = dataset

    def dataloaders(self, batch_size, num_workers, shuffle_train, shuffle_val):
        # discard metadata
        # and put num_channels (num_voices) at the last dimension
        return [({'x': t[0].transpose(1, 2)}
                 for t in dataloader)
                for dataloader
                in self.dataset.data_loaders(batch_size, num_workers=num_workers,
                                             shuffle_train=shuffle_train,
                                             shuffle_val=shuffle_val
                                             )]

    def write(self, x, path):
        """
        :param x: (batch_size, num_events, num_channels)
        :return: list of music21.Score
        """
        score = self.dataset.tensor_to_score(x.transpose(1, 0))
        score.write('xml', f'{path}.xml')
        return score

    def to_score(self, x):
        score = self.dataset.tensor_to_score(x.transpose(1, 0))
        return score
