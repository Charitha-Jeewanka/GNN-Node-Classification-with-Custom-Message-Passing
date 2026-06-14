# GNN Node Classification Cookbook

This cookbook provides step-by-step instructions to reproduce the virtual environment setup, run experiments for all GNN architectures, and generate the comparative performance and clustering visualizations.

---

## 1. Prerequisites & Environment Setup

This project uses the Python package manager `uv` for environment management.

### Step 1: Install `uv`
If `uv` is not installed on your system, install it via the official installer:
```powershell
# On Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```

### Step 2: Initialize Virtual Environment & Sync Dependencies
Navigate to the project directory and run the sync command. This will automatically create a `.venv` directory, install Python 3.12 (if needed), install PyTorch with CUDA support, and sync all packages listed in `requirements.txt`:
```bash
# Sync environment
uv sync
```

---

## 2. Running Subgraph Visualizations

To extract and visualize a connected neighborhood subgraph of the Cora dataset using Breadth-First Search (BFS):
```bash
uv run python main.py --mode visualize
```
*   **Configuration**: You can adjust the starting `seed_node` and number of `subgraph_nodes` under the `visualization` block in `config.yaml`.
*   **Output**: The plot is saved to `outputs/cora_subgraph.png`.

---

## 3. Running Single Model Training Runs

You can train any GNN configuration individually by modifying `config.yaml` or running `main.py`.

### GCN Baseline
1. Set the model type in `config.yaml`:
   ```yaml
   model:
     type: "GCN"
   ```
2. Execute the training run:
   ```bash
   uv run python main.py --mode train
   ```

### GraphSAGE (Mean, Max, or LSTM)
1. Configure GraphSAGE and select the aggregator in `config.yaml`:
   ```yaml
   model:
     type: "GraphSAGE"
     sage_aggregator: "mean"  # Change to "max" or "lstm"
   ```
2. Execute the training run:
   ```bash
   uv run python main.py --mode train
   ```

### Custom Attention GNN (Single-Head GAT)
1. Configure the attention model type in `config.yaml`:
   ```yaml
   model:
     type: "CustomAttention"
   ```
2. Execute the training run:
   ```bash
   uv run python main.py --mode train
   ```

All single training runs automatically:
- Save log metrics in `gnn_project.log`
- Save the best validation weights checkpoint under `checkpoints/` (e.g. `checkpoints/best_gcn_model.pth` or `checkpoints/best_graphsage_mean_model.pth`)
- Generate loss and accuracy curves (saved to `outputs/`)

---

## 4. Running the Systematic Experiment Suite

To train all GNN model configurations (GCN, GraphSAGE mean/max/lstm, CustomAttention) in sequence, save separate model checkpoints, and produce a comparative results summary:
```bash
uv run python run_experiments.py
```
*   **Resource Management**: This script pauses for 1.5 seconds between runs to allow CUDA contexts to clear, preventing GPU driver crashes.
*   **Outputs**:
    *   Saves final metrics to `outputs/experiment_results.json`.
    *   Prints a formatted Markdown-style results table directly to the console.
    *   Populates `checkpoints/` with distinct weights files.

---

## 5. Generating Comparative UMAP Plots and Metrics Chart

Once all experiments have finished and the model checkpoints are saved in `checkpoints/`, run the visualization script:
```bash
uv run python run_visualization.py
```
This script:
1.  Loads the best checkpoint weights for each GNN model.
2.  Passes the Cora dataset features through the models to extract the first-layer $64$-dimensional embeddings.
3.  Performs UMAP dimensionality reduction to compress the raw $1,433$-dim features and all GNN embeddings down to 2D.
4.  Generates a $2 \times 3$ grid comparison plot saved to `outputs/embedding_clusters.png`.
5.  Generates a dual-axis comparative metrics bar chart plotting Test Accuracy vs Test Loss saved to `outputs/performance_comparison.png`.
