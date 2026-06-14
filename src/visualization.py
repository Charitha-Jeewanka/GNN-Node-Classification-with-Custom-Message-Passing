import logging
import os
from typing import List, Optional, Set

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for headless/non-gui environments
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

def plot_subgraph(
    data: Data,
    num_nodes: int = 50,
    seed_node: int = 0,
    output_path: str = "outputs/subgraph.png",
    num_classes: Optional[int] = None
) -> None:
    """Extracts a connected subgraph around a seed node using BFS,

    and plots it using NetworkX and Matplotlib. Nodes are colored by their
    ground-truth class label.
    """
    logger.info(
        f"Extracting subgraph of {num_nodes} nodes starting from seed node {seed_node}..."
    )
    
    edge_index = data.edge_index
    num_total_nodes = data.num_nodes

    # Validate seed node
    if seed_node < 0 or seed_node >= num_total_nodes:
        logger.warning(
            f"Seed node {seed_node} out of bounds. Using node 0."
        )
        seed_node = 0

    # Build adjacency list representation for BFS
    # Using directed or undirected is fine, Cora is undirected
    adj_list = {i: [] for i in range(num_total_nodes)}
    for i in range(edge_index.size(1)):
        u = edge_index[0, i].item()
        v = edge_index[1, i].item()
        adj_list[u].append(v)

    # Breadth-First Search (BFS) to gather nodes
    queue: List[int] = [seed_node]
    visited: Set[int] = {seed_node}

    while queue and len(visited) < num_nodes:
        curr = queue.pop(0)
        for neighbor in adj_list[curr]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                if len(visited) >= num_nodes:
                    break

    subgraph_nodes = sorted(list(visited))
    subgraph_nodes_set = visited

    # Construct the induced subgraph edges
    subgraph_edges: List[tuple] = []
    for i in range(edge_index.size(1)):
        u = edge_index[0, i].item()
        v = edge_index[1, i].item()
        if u in subgraph_nodes_set and v in subgraph_nodes_set:
            # Avoid duplicate undirected edges in NetworkX
            if u < v:
                subgraph_edges.append((u, v))

    # Initialize NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(subgraph_nodes)
    G.add_edges_from(subgraph_edges)

    # Get class labels for the subgraph nodes
    y_sub = data.y[subgraph_nodes].cpu().numpy()
    
    # Retrieve the color map
    max_class_id = int(data.y.max().item())
    total_classes = num_classes if num_classes is not None else (max_class_id + 1)
    
    # Select tab10 or tab20 depending on number of classes
    cmap_name = "tab10" if total_classes <= 10 else "tab20"
    cmap = plt.get_cmap(cmap_name)

    # Build node color list
    node_colors = [cmap(label % 20) for label in y_sub]

    # Plot setup
    plt.figure(figsize=(10, 8), dpi=150)
    
    # Use spring layout for node placement
    pos = nx.spring_layout(G, seed=42, k=0.15, iterations=50)

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=300,
        edgecolors="#222222",
        linewidths=1.0,
        alpha=0.9
    )

    # Draw edges
    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.4,
        width=1.0,
        edge_color="#888888"
    )

    # Draw labels (node IDs) inside nodes
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=6,
        font_color="#ffffff",
        font_weight="bold"
    )

    # Create legend handles based on visible classes in the subgraph
    unique_sub_labels = sorted(list(set(y_sub.tolist())))
    legend_handles = []
    for label in unique_sub_labels:
        legend_handles.append(
            plt.Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor=cmap(label % 20),
                markersize=10,
                markeredgecolor="#222222",
                label=f"Class {label}"
            )
        )
    
    plt.legend(handles=legend_handles, loc="upper right", frameon=True)
    plt.title(
        f"GNN Citation Network Subgraph\n(Nodes: {len(subgraph_nodes)}, Seed Node: {seed_node})",
        fontsize=12,
        fontweight="bold",
        pad=10
    )
    plt.axis("off")

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the figure
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Successfully saved subgraph visualization to '{output_path}'.")

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    output_path: str = "outputs/training_curves.png"
) -> None:
    """Plots training and validation loss and accuracy curves side-by-side."""
    logger.info(
        f"Plotting training curves and saving to '{output_path}'..."
    )
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(14, 5), dpi=150)
    
    # Left Panel: Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", color="#1f77b4", linewidth=1.5)
    plt.plot(epochs, val_losses, label="Val Loss", color="#ff7f0e", linewidth=1.5)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.title("Loss Curves", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    
    # Right Panel: Accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train Accuracy", color="#2ca02c", linewidth=1.5)
    plt.plot(epochs, val_accs, label="Val Accuracy", color="#d62728", linewidth=1.5)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Accuracy", fontsize=10)
    plt.title("Accuracy Curves", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    
    plt.suptitle(
        "GNN Training and Validation Performance",
        fontsize=14,
        fontweight="bold"
    )
    plt.tight_layout()
    
    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Successfully saved training curves to '{output_path}'.")
