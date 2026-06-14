import json
import os
import re
import subprocess
import time
from typing import Any, Dict, List, Optional
import yaml

def run_experiment(model_type: str, sage_aggregator: Optional[str] = None) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    agg_suffix = f" ({sage_aggregator})" if sage_aggregator else ""
    print(f"Running Experiment: Model={model_type}{agg_suffix}")
    print("=" * 60)
    
    # Read base config.yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # Modify for this specific run
    config["model"]["type"] = model_type
    if sage_aggregator:
        config["model"]["sage_aggregator"] = sage_aggregator
        
    # Write to temporary config file
    temp_config_path = f"temp_config_{model_type.lower()}_{sage_aggregator or 'none'}.yaml"
    with open(temp_config_path, "w") as f:
        yaml.safe_dump(config, f)
        
    try:
        # Run main.py as a subprocess
        result = subprocess.run(
            ["uv", "run", "python", "main.py", "--config", temp_config_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Combine stdout and stderr since Python logging outputs to stderr by default
        output = result.stdout + "\n" + result.stderr
        
        # Extract performance metrics using regex
        best_val_match = re.search(r"Best Val Acc:\s*([0-9.]+)", output)
        best_epoch_match = re.search(r"at Epoch\s*([0-9]+)", output)
        test_acc_match = re.search(r"Final Test Accuracy of best model:\s*([0-9.]+)", output)
        test_loss_match = re.search(r"Test Loss:\s*([0-9.]+)", output)
        
        best_val = float(best_val_match.group(1)) if best_val_match else None
        best_epoch = int(best_epoch_match.group(1)) if best_epoch_match else None
        test_acc = float(test_acc_match.group(1)) if test_acc_match else None
        test_loss = float(test_loss_match.group(1)) if test_loss_match else None
        
        print(f"Success! Best Val Acc: {best_val} at epoch {best_epoch} | Test Acc: {test_acc} (Loss: {test_loss})")
        
        return {
            "Model": model_type,
            "Aggregator": sage_aggregator or ("Attention Gating" if model_type == "CustomAttention" else "N/A (Symmetric)"),
            "Best Val Acc": best_val,
            "Best Epoch": best_epoch,
            "Test Acc": test_acc,
            "Test Loss": test_loss,
            "Status": "Success"
        }
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print(f"Error Output:\n{e.stdout}\n{e.stderr}")
        return {
            "Model": model_type,
            "Aggregator": sage_aggregator or ("Attention Gating" if model_type == "CustomAttention" else "N/A (Symmetric)"),
            "Best Val Acc": None,
            "Best Epoch": None,
            "Test Acc": None,
            "Test Loss": None,
            "Status": f"Failed (Exit Code {e.returncode})"
        }
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return {
            "Model": model_type,
            "Aggregator": sage_aggregator or ("Attention Gating" if model_type == "CustomAttention" else "N/A (Symmetric)"),
            "Best Val Acc": None,
            "Best Epoch": None,
            "Test Acc": None,
            "Test Loss": None,
            "Status": f"Failed: {str(e)}"
        }
    finally:
        # Clean up temporary config file
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

def main() -> None:
    # 1. Define experiments
    experiments = [
        {"model": "GCN", "aggregator": None},
        {"model": "GraphSAGE", "aggregator": "mean"},
        {"model": "GraphSAGE", "aggregator": "max"},
        {"model": "GraphSAGE", "aggregator": "lstm"},
        {"model": "CustomAttention", "aggregator": None},
    ]
    
    results: List[Dict[str, Any]] = []
    
    # 2. Run all experiments
    for exp in experiments:
        res = run_experiment(exp["model"], exp["aggregator"])
        results.append(res)
        time.sleep(1.5)  # Allow CUDA context to tear down and free memory
        
    # 3. Print final comparative table
    print("\n" + "=" * 80)
    print("                    FINAL EXPERIMENT RESULTS COMPARISON")
    print("=" * 80)
    header = f"{'Model':<12} | {'Aggregator':<15} | {'Best Val Acc':<12} | {'Best Epoch':<10} | {'Test Acc':<10} | {'Test Loss':<10}"
    print(header)
    print("-" * 80)
    for r in results:
        if r["Status"] == "Success":
            val_acc = f"{r['Best Val Acc']:.4f}"
            test_acc = f"{r['Test Acc']:.4f}"
            test_loss = f"{r['Test Loss']:.4f}"
            epoch = f"{r['Best Epoch']}"
        else:
            val_acc = "N/A"
            test_acc = "N/A"
            test_loss = "N/A"
            epoch = "N/A"
            
        row = f"{r['Model']:<12} | {r['Aggregator']:<15} | {val_acc:<12} | {epoch:<10} | {test_acc:<10} | {test_loss:<10}"
        print(row)
    print("=" * 80)
    
    # 4. Save results to output directory
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/experiment_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Saved comparative results to 'outputs/experiment_results.json'.")

if __name__ == "__main__":
    main()
