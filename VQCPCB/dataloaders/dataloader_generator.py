class DataloaderGenerator:
    """
    Base abstract class for data loader generators
    dataloaders
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def dataloaders(self, batch_size, num_workers=0, shuffle_train=True,
                    shuffle_val=False):
        raise NotImplementedError

