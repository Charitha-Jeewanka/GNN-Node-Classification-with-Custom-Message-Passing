import json
import os
import logging
from typing import Dict, Any, List

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

from src.config import load_config
from src.dataset import load_planetoid_dataset
from src.models import GCN, GraphSAGE, CustomAttentionGNN

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("visualization")

def get_model_embeddings(model_type: str, sage_aggregator: str, config: Any, dataset: Any, data: Any, device: torch.device) -> np.ndarray:
    """Instantiates a model, loads the best saved checkpoint, and extracts node embeddings."""
    in_dim = dataset.num_features
    hidden_dim = config.model.hidden_dim
    out_dim = dataset.num_classes
    
    if model_type == "GCN":
        model = GCN(in_dim, hidden_dim, out_dim, config.model.dropout)
        weights_name = "best_gcn_model.pth"
    elif model_type == "GraphSAGE":
        model = GraphSAGE(in_dim, hidden_dim, out_dim, sage_aggregator, config.model.dropout)
        weights_name = f"best_graphsage_{sage_aggregator.lower()}_model.pth"
    elif model_type == "CustomAttention":
        model = CustomAttentionGNN(in_dim, hidden_dim, out_dim, config.model.dropout, attn_dropout=0.1)
        weights_name = "best_customattention_model.pth"
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    weights_path = os.path.join("checkpoints", weights_name)
    if not os.path.exists(weights_path):
        logger.error(f"Checkpoint not found at '{weights_path}'")
        raise FileNotFoundError(weights_path)
        
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        emb = model.get_embeddings(data.x.to(device), data.edge_index.to(device))
        
    return emb.cpu().numpy()

def main() -> None:
    # 1. Setup device and load config/dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    config = load_config("config.yaml")
    dataset = load_planetoid_dataset(config.dataset.name, config.dataset.root_dir)
    data = dataset[0]
    labels = data.y.cpu().numpy()
    
    # Define the 5 configurations and their labels
    configs = [
        {"type": "GCN", "agg": "none", "title": "GCN Baseline", "acc": 0.8150},
        {"type": "GraphSAGE", "agg": "mean", "title": "GraphSAGE (mean)", "acc": 0.8030},
        {"type": "GraphSAGE", "agg": "max", "title": "GraphSAGE (max)", "acc": 0.7800},
        {"type": "GraphSAGE", "agg": "lstm", "title": "GraphSAGE (lstm)", "acc": 0.7390},
        {"type": "CustomAttention", "agg": "none", "title": "CustomAttention (GAT)", "acc": 0.7990},
    ]
    
    # 2. Extract node embeddings for all configurations
    embeddings_dict = {}
    for c in configs:
        key = f"{c['type']}_{c['agg']}"
        logger.info(f"Extracting embeddings for {c['title']}...")
        embeddings_dict[key] = get_model_embeddings(c["type"], c["agg"], config, dataset, data, device)
        
    # Include raw input features as the 6th baseline
    logger.info("Extracting Raw input features baseline...")
    raw_features = data.x.cpu().numpy()
    embeddings_dict["Raw"] = raw_features
    
    # 3. Apply UMAP to reduce embeddings to 2D
    umap_2d_dict = {}
    for name, emb in embeddings_dict.items():
        logger.info(f"Running UMAP on {name} embeddings...")
        # Use a fixed random state for visualization consistency
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        umap_2d_dict[name] = reducer.fit_transform(emb)
        
    # 4. Plot UMAP embeddings in a 2x3 grid
    logger.info("Generating UMAP comparative cluster plot...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=150)
    axes = axes.ravel()
    
    cmap = plt.get_cmap("tab10")
    unique_labels = np.unique(labels)
    
    all_plot_configs = configs + [{"type": "Raw", "agg": "none", "title": "Raw Input Features (1433-dim)", "acc": 0.0}]
    
    for i, c in enumerate(all_plot_configs):
        ax = axes[i]
        key = "Raw" if c["type"] == "Raw" else f"{c['type']}_{c['agg']}"
        coords = umap_2d_dict[key]
        
        # Plot scatter
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=labels, cmap=cmap,
            s=4, alpha=0.7, edgecolors="none"
        )
        
        # Add labels
        if c["type"] == "Raw":
            ax.set_title(c["title"], fontsize=12, fontweight="bold", pad=8)
        else:
            ax.set_title(f"{c['title']}\nTest Accuracy: {c['acc']*100:.2f}%", fontsize=12, fontweight="bold", pad=8)
            
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        
    # Create unified legend for all subplots
    handles = []
    for label in unique_labels:
        handles.append(
            plt.Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=cmap(label % 10), markersize=8,
                label=f"Class {label}"
            )
        )
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=True, fontsize=10, bbox_to_anchor=(0.5, 0.04))
    
    plt.suptitle("UMAP Dimensionality Projections of Learned GNN Node Embeddings (Cora)", fontsize=16, fontweight="bold", y=0.96)
    plt.subplots_adjust(bottom=0.12, hspace=0.25, wspace=0.15)
    
    os.makedirs(config.visualization.output_dir, exist_ok=True)
    cluster_plot_path = os.path.join(config.visualization.output_dir, "embedding_clusters.png")
    plt.savefig(cluster_plot_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved cluster visualization to '{cluster_plot_path}'")
    
    # 5. Plot comparative performance metrics side-by-side
    logger.info("Generating comparative performance bar chart...")
    names = [c["title"] for c in configs]
    accuracies = [c["acc"] * 100 for c in configs]
    losses = [1.0971, 0.9091, 0.8928, 1.0495, 0.8953] # Extracted from logs
    
    x = np.arange(len(names))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=150)
    
    # Plot Accuracies (Left axis)
    color = "#1f77b4"
    ax1.set_xlabel("Model Configuration", fontweight="bold", labelpad=10)
    ax1.set_ylabel("Test Accuracy (%)", color=color, fontweight="bold")
    bars1 = ax1.bar(x - width/2, accuracies, width, label="Test Accuracy (%)", color=color, alpha=0.85, edgecolor="black", linewidth=0.7)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(60, 85)
    ax1.grid(True, linestyle="--", alpha=0.3, axis="y")
    
    # Create second axis for losses
    ax2 = ax1.twinx()
    color = "#ff7f0e"
    ax2.set_ylabel("Test Loss", color=color, fontweight="bold")
    bars2 = ax2.bar(x + width/2, losses, width, label="Test Loss", color=color, alpha=0.85, edgecolor="black", linewidth=0.7)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(0.5, 1.2)
    
    # Add labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f"{height:.2f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color="#1f77b4", fontweight="bold")
                    
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f"{height:.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color="#ff7f0e", fontweight="bold")
                    
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    
    # Unified legend
    lines = [bars1, bars2]
    labels_legend = [line.get_label() for line in lines]
    ax1.legend(lines, labels_legend, loc="upper right")
    
    plt.title("GNN Model Performance Metrics Comparison (Cora)", fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    
    comparison_plot_path = os.path.join(config.visualization.output_dir, "performance_comparison.png")
    plt.savefig(comparison_plot_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved comparative bar chart to '{comparison_plot_path}'")
    
    logger.info("Milestone 5 visualization scripts complete!")

if __name__ == "__main__":
    main()
