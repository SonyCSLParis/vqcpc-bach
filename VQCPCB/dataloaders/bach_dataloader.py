import music21
from VQCPCB.datasets.chorale_dataset import ChoraleBeatsDataset
from VQCPCB.dataloaders.dataloader_generator import DataloaderGenerator

subdivision = 4
num_voices = 4


class BachDataloaderGenerator(DataloaderGenerator):
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

    def dataloaders(self, batch_size, num_workers=0, shuffle_train=True,
                    shuffle_val=False):
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
