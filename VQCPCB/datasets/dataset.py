import os
from abc import ABC, abstractmethod
import VQCPCB


class Dataset(ABC):
    def __init__(self):
        self.database_root = os.path.dirname(VQCPCB.__file__)
        return

    @abstractmethod
    def get_dataset(self):
        pass
