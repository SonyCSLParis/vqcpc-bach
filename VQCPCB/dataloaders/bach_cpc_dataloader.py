import torch
import music21
from VQCPCB.dataloaders.cpc_dataloader import CPCDataloaderGenerator
from VQCPCB.datasets.chorale_dataset import ChoraleBeatsDataset

subdivision = 4
num_voices = 4


class BachCPCDataloaderGenerator(CPCDataloaderGenerator):
    def __init__(self,
                 num_tokens_per_block,
                 num_blocks_left,
                 num_blocks_right,
                 negative_sampling_method,
                 num_negative_samples,
                 *args, **kwargs):
        """

        :param num_tokens_per_block:
        :param num_blocks_left:
        :param num_blocks_right:
        :param num_negative_samples:
        :param negative_sampling_method:
        :param args:
        :param kwargs:
        """
        assert num_tokens_per_block % (subdivision * num_voices) == 0
        super(BachCPCDataloaderGenerator, self).__init__(
            num_tokens_per_block,
            num_blocks_left,
            num_blocks_right,
            negative_sampling_method,
            num_negative_samples)
        # load dataset
        datasets = self._dataset()
        self.dataset_positive = datasets['positive']
        self.dataset_negative = datasets['negative']
        self.num_channels = num_voices

    def _dataset(self):
        """
        Loads the appropriate dataset depending on the sampling method
        :return: ChoraleDataset or tuple(ChoraleDataset)
        """
        if self.negative_sampling_method == 'same_sequence':
            num_tokens_per_beat = subdivision * num_voices
            num_tokens = self.num_tokens_per_block * (self.num_blocks_left + self.num_blocks_right)

            assert num_tokens % num_tokens_per_beat == 0

            # Positive dataset
            num_beats_positive = num_tokens // num_tokens_per_beat
            dataset = ChoraleBeatsDataset(
                corpus_it_gen=music21.corpus.chorales.Iterator,
                voice_ids=[0, 1, 2, 3],
                metadatas=[],
                sequences_size=num_beats_positive,
                subdivision=subdivision,
            )
            return dict(positive=dataset, negative=None)

        elif self.negative_sampling_method in ['random']:
            num_tokens_per_beat = subdivision * num_voices
            num_tokens = self.num_tokens_per_block * (self.num_blocks_left + self.num_blocks_right)

            assert num_tokens % num_tokens_per_beat == 0

            # Positive dataset
            num_beats_positive = num_tokens // num_tokens_per_beat
            dataset_positive = ChoraleBeatsDataset(
                corpus_it_gen=music21.corpus.chorales.Iterator,
                voice_ids=[0, 1, 2, 3],
                metadatas=[],
                sequences_size=num_beats_positive,
                subdivision=subdivision,
            )

            num_tokens_per_beat = subdivision * num_voices
            num_beats_negative = self.num_tokens_per_block // num_tokens_per_beat
            dataset_negative = ChoraleBeatsDataset(
                corpus_it_gen=music21.corpus.chorales.Iterator,
                voice_ids=[0, 1, 2, 3],
                metadatas=[],
                sequences_size=num_beats_negative,
                subdivision=subdivision,
            )
            return dict(positive=dataset_positive, negative=dataset_negative)
        else:
            raise NotImplementedError

    def dataloaders(self,
                    batch_size,
                    num_workers=0
                    ):
        """

        :return: torch Dataloader, returns a dict of
        """
        #
        if self.negative_sampling_method == 'random':
            return self._dataloader_random(batch_size=batch_size,
                                           num_workers=num_workers)
        elif self.negative_sampling_method == 'same_sequence':
            return self._dataloader_same_sequence(batch_size=batch_size,
                                                  num_workers=num_workers)
        else:
            raise NotImplementedError

    def _dataloader_same_sequence(self, batch_size, num_workers):
        """
        Dataloader for negative_sampling_method == 'random'
        :param batch_size:
        :return: For all dataloaders, return type must be
            x_left: (batch_size, num_events_left, num_channels)
            x_right: (batch_size, num_events_right, num_channels)
            x_negative_samples: (batch_size, num_negative_samples, fks_dim, num_events_per_block, num_channels)
        """
        # dataset should be initialized by self._dataset
        # WARNING self.num_negative_samples parameter is not used
        num_negative_samples = self.num_blocks_right + self.num_blocks_left - 1
        assert self.dataset_positive is not None
        num_tokens_left = self.num_tokens_per_block * self.num_blocks_left

        dataloaders = self.dataset_positive.data_loaders(
            batch_size=batch_size,
            num_workers=num_workers
        )

        # Generate dataloaders
        def _aggregate_dataloader(dataloader):
            for p in dataloader:
                # remove metadata
                p = p[0]

                x_left = p[:, :, :num_tokens_left // num_voices]
                x_right = p[:, :, num_tokens_left // num_voices:]

                # generate negative samples
                negative_sample = self._build_negatives_sameSeq(x_left, x_right, batch_size, num_negative_samples)
                negative_sample_back = self._build_negatives_sameSeq(x_right, x_left, batch_size, num_negative_samples)
                x = {
                    'x_left': x_left.transpose(1, 2),
                    'x_right': x_right.transpose(1, 2),
                    'negative_samples': negative_sample.transpose(3, 4)
                    'negative_samples_back': negative_sample_back.transpose(3, 4)
                }

                yield x

        dataloaders = [
            _aggregate_dataloader(dataloader)
            for dataloader
            in dataloaders
        ]

        return dataloaders

    def _build_negatives_sameSeq(self, x_left, x_right, batch_size, num_negative_samples):
        negative_sample = []
        for k in range(self.num_blocks_right):
            x_right_split = x_right.unsqueeze(1).split(self.num_tokens_per_block //
                                                       num_voices,
                                                       dim=3)
            neg_k = torch.cat(
                (*x_left.unsqueeze(1).split(self.num_tokens_per_block // num_voices, dim=3),
                 *x_right_split[:k],
                 *x_right_split[k + 1:]),
                dim=1
            ).unsqueeze(2)
            negative_sample.append(neg_k)
        negative_sample = torch.cat(negative_sample, dim=2)

        negative_sample = negative_sample.view(
            batch_size,
            num_negative_samples,
            self.num_blocks_right,
            num_voices,
            self.num_tokens_per_block // num_voices
        )
        return negative_sample

    def _dataloader_random(self, batch_size, num_workers):
        """
        Dataloader for negative_sampling_method == 'random'
        :param batch_size:
        :return:
            x_left: (batch_size, num_events_left, num_channels)
            x_right: (batch_size, num_events_right, num_channels)
            x_negative_samples: (batch_size, num_negative_samples, fks_dim, num_events_per_block, num_channels)
        """

        # dataset should be initialized by self._dataset
        assert self.dataset_positive is not None
        num_tokens_left = self.num_tokens_per_block * self.num_blocks_left

        positive_dataloaders = self.dataset_positive.data_loaders(
            batch_size=batch_size,
            num_workers=num_workers,
            indexed_dataloaders=False
        )

        # Negative dataset
        negative_dataloaders = self.dataset_negative.data_loaders(
            batch_size=batch_size * self.num_negative_samples * self.num_blocks_right,
            num_workers=num_workers,
            indexed_dataloaders=False
        )

        # Generate dataloaders
        def _aggregate_dataloader(dataloader_positive,
                                  dataloader_negative):
            for p, n in zip(dataloader_positive, dataloader_negative):
                # remove metadata
                negative_sample = n[0]
                p = p[0]
                assert negative_sample.size(2) == self.num_tokens_per_block // num_voices
                negative_sample = negative_sample.view(
                    batch_size,
                    self.num_negative_samples,
                    self.num_blocks_right,
                    num_voices,
                    self.num_tokens_per_block // num_voices
                )
                x_left = p[:, :, :num_tokens_left // num_voices]
                x_right = p[:, :, num_tokens_left // num_voices:]

                # Transpose is to put channel (=voice) dim last
                x = {
                    'x_left': x_left.transpose(1, 2),
                    'x_right': x_right.transpose(1, 2),
                    'negative_samples': negative_sample.transpose(3, 4),
                    'negative_samples_back': negative_sample.transpose(3, 4)
                }

                yield x

        dataloaders = [
            _aggregate_dataloader(dataloader_positive, dataloader_negative)
            for dataloader_positive, dataloader_negative
            in zip(positive_dataloaders, negative_dataloaders)
        ]

        return dataloaders