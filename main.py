import argparse
import logging
import os
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn

from src.config import load_config
from src.dataset import load_planetoid_dataset
from src.engine import evaluate, train_epoch
from src.models import GCN, GraphSAGE, CustomAttentionGNN
from src.visualization import plot_subgraph, plot_training_curves

# Configure logging to console and a log file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("gnn_project.log", mode="a")
    ]
)
logger = logging.getLogger("main")

def set_seed(seed: int) -> None:
    """Sets random seeds across random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GNN Message Passing Project - Node Classification"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["visualize", "train"],
        help="Mode: visualize a subgraph or train a model (default: train)"
    )
    args = parser.parse_args()

    # Load configuration
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found at '{args.config}'")
        return

    config = load_config(args.config)
    
    # Set seed for reproducibility
    set_seed(config.training.seed)
    
    # Load dataset
    dataset = load_planetoid_dataset(
        name=config.dataset.name,
        root_dir=config.dataset.root_dir
    )
    data = dataset[0]
    
    # Device configuration
    device = torch.device("cuda" if config.training.device == "cuda" and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Send data to device
    data = data.to(device)
    
    if args.mode == "visualize":
        # Output path for visualization
        output_filename = f"{config.dataset.name.lower()}_subgraph.png"
        output_path = os.path.join(config.visualization.output_dir, output_filename)
        
        # Generate and save the subgraph visualization
        plot_subgraph(
            data=data,
            num_nodes=config.visualization.subgraph_nodes,
            seed_node=config.visualization.seed_node,
            output_path=output_path,
            num_classes=dataset.num_classes
        )
        logger.info("Visualization mode complete.")
        return

    # Train Mode
    logger.info(f"Initializing model of type: {config.model.type}")
    
    if config.model.type == "GCN":
        model = GCN(
            in_channels=dataset.num_features,
            hidden_dim=config.model.hidden_dim,
            out_channels=dataset.num_classes,
            dropout=config.model.dropout
        )
    elif config.model.type == "GraphSAGE":
        model = GraphSAGE(
            in_channels=dataset.num_features,
            hidden_dim=config.model.hidden_dim,
            out_channels=dataset.num_classes,
            aggregator_type=config.model.sage_aggregator,
            dropout=config.model.dropout
        )
    elif config.model.type == "CustomAttention":
        model = CustomAttentionGNN(
            in_channels=dataset.num_features,
            hidden_dim=config.model.hidden_dim,
            out_channels=dataset.num_classes,
            dropout=config.model.dropout,
            attn_dropout=0.1  # Set 10% attention dropout for regularization
        )
    else:
        logger.error(f"Model type '{config.model.type}' is not supported.")
        return

    model = model.to(device)
    logger.info(model)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    train_losses: List[float] = []
    val_losses: List[float] = []
    train_accs: List[float] = []
    val_accs: List[float] = []

    best_val_acc = 0.0
    best_epoch = 0

    logger.info("Starting training loop...")
    for epoch in range(1, config.training.epochs + 1):
        loss = train_epoch(model, data, optimizer, criterion)
        
        train_loss, val_loss, test_loss, train_acc, val_acc, test_acc = evaluate(
            model, data, criterion
        )
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Track best validation performance
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            # Save best model weights
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                model.state_dict(),
                f"checkpoints/best_{config.model.type.lower()}_model.pth"
            )

        if epoch == 1 or epoch % 10 == 0 or epoch == config.training.epochs:
            logger.info(
                f"Epoch: {epoch:03d} | Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}"
            )

    logger.info(f"Training completed. Best Val Acc: {best_val_acc:.4f} at Epoch {best_epoch}.")
    
    # Load best model weights and evaluate final test performance
    best_weights_path = f"checkpoints/best_{config.model.type.lower()}_model.pth"
    if os.path.exists(best_weights_path):
        model.load_state_dict(torch.load(best_weights_path, weights_only=True))
        logger.info(f"Loaded best model weights from '{best_weights_path}'")
        
    _, _, test_loss, _, _, test_acc = evaluate(model, data, criterion)
    logger.info(f"Final Test Accuracy of best model: {test_acc:.4f} (Test Loss: {test_loss:.4f})")

    # Save training curves plot
    if config.model.type == "GraphSAGE":
        curves_filename = f"graphsage_{config.model.sage_aggregator.lower()}_training_curves.png"
    else:
        curves_filename = f"{config.model.type.lower()}_training_curves.png"
    curves_path = os.path.join(config.visualization.output_dir, curves_filename)
    plot_training_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        train_accs=train_accs,
        val_accs=val_accs,
        output_path=curves_path
    )
    
    logger.info(f"Training and evaluation of {config.model.type} completed successfully.")

if __name__ == "__main__":
    main()
