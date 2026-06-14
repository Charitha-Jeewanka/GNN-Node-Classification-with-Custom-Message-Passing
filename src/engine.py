import logging
from typing import Tuple

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

def train_epoch(
    model: Module,
    data: Data,
    optimizer: Optimizer,
    criterion: Module
) -> float:
    """Trains the GNN model for one epoch over the training nodes.

    Returns the training loss value.
    """
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    out = model(data.x, data.edge_index)
    
    # Compute training loss
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    
    # Backward pass and gradient step
    loss.backward()
    optimizer.step()
    
    return float(loss.item())

@torch.no_grad()
def evaluate(
    model: Module,
    data: Data,
    criterion: Module
) -> Tuple[float, float, float, float, float, float]:
    """Evaluates the GNN model across train, validation, and test splits.

    Returns:
        A tuple of:
        (train_loss, val_loss, test_loss, train_acc, val_acc, test_acc)
    """
    model.eval()
    out = model(data.x, data.edge_index)
    
    # Calculate losses
    train_loss = float(criterion(out[data.train_mask], data.y[data.train_mask]).item())
    val_loss = float(criterion(out[data.val_mask], data.y[data.val_mask]).item())
    test_loss = float(criterion(out[data.test_mask], data.y[data.test_mask]).item())
    
    # Calculate accuracy
    preds = out.argmax(dim=-1)
    
    train_correct = (preds[data.train_mask] == data.y[data.train_mask]).sum().item()
    train_total = int(data.train_mask.sum().item())
    train_acc = train_correct / train_total if train_total > 0 else 0.0
    
    val_correct = (preds[data.val_mask] == data.y[data.val_mask]).sum().item()
    val_total = int(data.val_mask.sum().item())
    val_acc = val_correct / val_total if val_total > 0 else 0.0
    
    test_correct = (preds[data.test_mask] == data.y[data.test_mask]).sum().item()
    test_total = int(data.test_mask.sum().item())
    test_acc = test_correct / test_total if test_total > 0 else 0.0
    
    return train_loss, val_loss, test_loss, train_acc, val_acc, test_acc
