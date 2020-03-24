import os
from abc import abstractmethod, ABC

import VQCPCB


class Dataset(ABC):
    def __init__(self):
        self.database_root = os.path.abspath(f'{os.path.dirname(VQCPCB.__file__)}/../data')
        if not os.path.isdir(self.database_root):
            os.mkdir(self.database_root)
        return

    @abstractmethod
    def data_loaders(self, batch_size, num_workers, indexed_dataloaders):
        pass
