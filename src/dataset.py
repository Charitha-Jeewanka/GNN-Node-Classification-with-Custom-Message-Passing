import logging
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

logger = logging.getLogger(__name__)

def load_planetoid_dataset(name: str, root_dir: str) -> Planetoid:
    """Loads a Planetoid dataset (Cora, PubMed, Citeseer) using PyTorch Geometric.

    Applies NormalizeFeatures transform by default.
    """
    logger.info(f"Loading Planetoid dataset '{name}' from root directory '{root_dir}'...")
    
    # Use T.NormalizeFeatures() to row-normalize bag-of-words node features
    dataset = Planetoid(root=root_dir, name=name, transform=T.NormalizeFeatures())
    
    data = dataset[0]
    logger.info("Dataset loading completed.")
    logger.info(f"Dataset name: {name}")
    logger.info(f"Number of graphs: {len(dataset)}")
    logger.info(f"Number of features: {dataset.num_features}")
    logger.info(f"Number of classes: {dataset.num_classes}")
    logger.info(f"Number of nodes: {data.num_nodes}")
    logger.info(f"Number of edges: {data.num_edges}")
    logger.info(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")
    logger.info(f"Number of training nodes: {data.train_mask.sum().item()}")
    logger.info(f"Number of validation nodes: {data.val_mask.sum().item()}")
    logger.info(f"Number of test nodes: {data.test_mask.sum().item()}")
    logger.info(f"Has isolated nodes: {data.has_isolated_nodes()}")
    logger.info(f"Has self-loops: {data.has_self_loops()}")
    logger.info(f"Is undirected: {data.is_undirected()}")
    
    return dataset
