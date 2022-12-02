from typing import Tuple
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, KarateClub, Reddit

from .preprocessing import remove_infrequent_classes
from .preprocessing import train_test_split

def load_dataset(name, test_ratio) -> Tuple[Data, Data]:
    """Downloads the requested dataset into the datasets/downloads/ directory,
    and returns a train/test split.
    
    Parameters
    ----------
    name : str
        one of ("amazon", "reddit", "karateclub")

    Returns
    -------
    
    """
    root = "datasets/download/"
    if name.lower() == "amazon":
        data = Amazon(root, "Computers")[0]
        class_threshold = 1000
    elif name.lower() == "reddit":
        data = Reddit(root)[0]
        class_threshold = 10000
    elif name.lower() == "karateclub":
        data = KarateClub()[0]
        class_threshold = 0
    
    data = remove_infrequent_classes(data, class_threshold)
    num_classes = len(data.y.unique())
    train, test = train_test_split(data, test_ratio)
    return train, test, num_classes
