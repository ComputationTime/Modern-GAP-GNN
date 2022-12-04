from typing import Tuple
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, KarateClub, Reddit

from .preprocessing import remove_infrequent_classes
from .preprocessing import train_test_split

def load_dataset(name, test_ratio, device="cpu") -> Tuple[Data, Data]:
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
        data = Amazon(root, "Computers")[0].to(device)
        class_threshold = 1000
    elif name.lower() == "reddit":
        data = Reddit(root)[0].to(device)
        class_threshold = 10000
    elif name.lower() == "karateclub":
        data = KarateClub()[0].to(device)
        class_threshold = 0
    
    data = remove_infrequent_classes(data, class_threshold, device)
    num_classes = len(data.y.unique())
    train, test = train_test_split(data, test_ratio, device)
    train.train_mask = torch.ones(train.x.size(dim=0), dtype=torch.bool)
    return train, test, num_classes
