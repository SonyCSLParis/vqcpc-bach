import os
import VQCPCB


class Dataset:
    def __init__(self):
        self.database_root = os.path.abspath(f'{os.path.dirname(VQCPCB.__file__)}/../data')
        if not os.path.isdir(self.database_root):
            os.mkdir(self.database_root)
        return
