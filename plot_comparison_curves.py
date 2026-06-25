import os
import json
import matplotlib.pyplot as plt

LOG_DIR = "training_logs"
OUTPUT_DIR = "evaluation_results"

KEY_EXPERIMENTS = {
    "Baseline CNN - Mel (Config A)": "cnn_config_A_history.json",
    "Baseline CNN - Stacked (Config D)": "cnn_config_D_history.json",
    "Proposed CNN-LSTM - Mel (Config A)": "hybrid_config_A_history.json",
    "Proposed CNN-LSTM - Stacked (Config D)": "hybrid_config_D_history.json"
}

def plot_comparison():
    print("Generating validation curve comparison plot...")
    
    plt.figure(figsize=(10, 6))
    
    colors = ['blue', 'orange', 'green', 'red']
    linestyles = ['--', '--', '-', '-']
    
    for i, (label, filename) in enumerate(KEY_EXPERIMENTS.items()):
        filepath = os.path.join(LOG_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Skipping...")
            continue
            
        with open(filepath, 'r') as f:
            history = json.load(f)
            
        val_score = [s * 100 for s in history['val_score']]
        epochs = range(1, len(val_score) + 1)
        
        plt.plot(
            epochs, 
            val_score, 
            label=label, 
            color=colors[i], 
            linestyle=linestyles[i],
            linewidth=2,
            alpha=0.85
        )
        
    plt.title("Validation ICBHI Score Convergence Comparison", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Validation ICBHI Score (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10, loc='lower right')
    plt.tight_layout()
    
    # Save to evaluation results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "validation_score_comparison.png")
    plt.savefig(save_path, dpi=300)
    print(f"Validation comparison plot successfully saved to {save_path}")
    
    # Save to artifacts directory
    artifact_dir = r"D:\Internship '26\Lung Disease>"
    if os.path.exists(artifact_dir):
        plt.savefig(os.path.join(artifact_dir, "validation_score_comparison.png"), dpi=300)
        print("Comparison plot copied to artifacts directory.")
        
    plt.close()

if __name__ == "__main__":
    plot_comparison()
