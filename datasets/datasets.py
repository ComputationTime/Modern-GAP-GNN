import torch
from torch_geometric.datasets import Amazon, KarateClub, Reddit, FacebookPagePage

from .preprocessing import remove_infrequent_classes
from .preprocessing import train_test_split

def load_dataset(name, test_ratio):
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
        data = Amazon(root + "Amazon", "Computers")[0]
        class_threshold = 1000
    elif name.lower() == "reddit":
        data = Reddit(root)[0]
        class_threshold = 10000
    elif name.lower() == "facebook":
        data = FacebookPagePage(root + "FacebookPagePage")[0]
        class_threshold = None
    elif name.lower() == "karateclub":
        data = KarateClub()[0]
        class_threshold = None
    else:
        raise NotImplementedError(f"Unknown dataset {name}")
    
    if class_threshold is not None:
        data = remove_infrequent_classes(data, class_threshold)
    num_classes = len(data.y.unique())
    train, test = train_test_split(data, test_ratio)
    train.train_mask = torch.ones(train.x.size(dim=0), dtype=torch.bool)

    return train, test, num_classes
