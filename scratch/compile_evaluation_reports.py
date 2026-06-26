import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Import dataset loader to get true ground truth labels
import sys
sys.path.append(os.path.abspath('.'))
from src.sota.dataset import get_dataloaders

def main():
    reports_dir = "evaluation_reports"
    results_dir = "evaluation_results"
    os.makedirs(reports_dir, exist_ok=True)
    
    print("Loading test dataset labels directly from dataloaders...")
    # Load standard dataloader (full test set)
    _, _, test_loader_std = get_dataloaders(multitask=False)
    cycle_labels_std = test_loader_std.dataset.get_labels()
    
    # Load multitask dataloader (filtered test set)
    _, _, test_loader_mt = get_dataloaders(multitask=True)
    cycle_labels_mt = test_loader_mt.dataset.get_labels()
    pathology_labels_mt = test_loader_mt.dataset.df['pathology_label'].values
    
    class_map = {0: "Normal", 1: "Crackles", 2: "Wheezes", 3: "Both"}
    pathology_map = {0: "COPD", 1: "URTI", 2: "Healthy"}
    
    # Files to process
    files_info = [
        {
            "filename": "hybrid_config_D_resnet_sota_results.json",
            "backbone": "ResNet-18",
            "task": "Single-Task",
            "labels": cycle_labels_std,
            "multitask": False
        },
        {
            "filename": "hybrid_config_D_resnet_sota_multitask_results.json",
            "backbone": "ResNet-18",
            "task": "Multi-Task",
            "labels": cycle_labels_mt,
            "multitask": True,
            "pathology_labels": pathology_labels_mt
        },
        {
            "filename": "hybrid_config_D_sota_results.json",
            "backbone": "PANNs Cnn14",
            "task": "Single-Task",
            "labels": cycle_labels_std,
            "multitask": False
        },
        {
            "filename": "hybrid_config_D_sota_multitask_results.json",
            "backbone": "PANNs Cnn14",
            "task": "Multi-Task",
            "labels": cycle_labels_mt,
            "multitask": True,
            "pathology_labels": pathology_labels_mt
        }
    ]
    
    summary_rows = []
    
    for info in files_info:
        path = os.path.join(results_dir, info["filename"])
        if not os.path.exists(path):
            print(f"Skipping missing file: {path}")
            continue
            
        print(f"Processing: {info['filename']}...")
        with open(path, 'r') as f:
            res = json.load(f)
            
        latency = res.get("latency_ms", 0.0)
        
        # Standard summary
        std = res["std"]
        summary_rows.append({
            "Backbone": info["backbone"],
            "Task": info["task"],
            "Calibration": "Standard (Argmax)",
            "Accuracy": f"{std['accuracy']*100:.2f}%",
            "Sensitivity (Se)": f"{std['se']*100:.2f}%",
            "Specificity (Sp)": f"{std['sp']*100:.2f}%",
            "ICBHI Score": f"{std['score']*100:.2f}%",
            "Pathology Acc": f"{res['pathology']['accuracy']*100:.2f}%" if info["multitask"] and "pathology" in res else "—",
            "Inference Latency": f"{latency:.2f} ms/cycle"
        })
        
        # Calibrated summary
        cal = res.get("calibrated")
        if cal:
            summary_rows.append({
                "Backbone": info["backbone"],
                "Task": info["task"],
                "Calibration": "Calibrated (Product)",
                "Accuracy": f"{cal['accuracy']*100:.2f}%",
                "Sensitivity (Se)": f"{cal['se']*100:.2f}%",
                "Specificity (Sp)": f"{cal['sp']*100:.2f}%",
                "ICBHI Score": f"{cal['score']*100:.2f}%",
                "Pathology Acc": f"{res['pathology']['accuracy']*100:.2f}%" if info["multitask"] and "pathology" in res else "—",
                "Inference Latency": f"{latency:.2f} ms/cycle"
            })
            
        # Detailed prediction CSV
        detailed_rows = []
        true_labels = info["labels"]
        std_preds = std["preds"]
        cal_preds = cal["preds"] if cal else None
        
        for idx in range(len(true_labels)):
            row = {
                "Cycle Index": idx,
                "Ground Truth Class": class_map[true_labels[idx]],
                "Argmax Prediction": class_map[std_preds[idx]],
                "Argmax Match": true_labels[idx] == std_preds[idx]
            }
            if cal_preds:
                row["Calibrated Prediction"] = class_map[cal_preds[idx]]
                row["Calibrated Match"] = true_labels[idx] == cal_preds[idx]
                
            if info["multitask"] and "pathology" in res:
                path_labels = info["pathology_labels"]
                path_preds = res["pathology"]["preds"]
                row["Ground Truth Pathology"] = pathology_map[path_labels[idx]]
                row["Predicted Pathology"] = pathology_map[path_preds[idx]]
                row["Pathology Match"] = path_labels[idx] == path_preds[idx]
                
            detailed_rows.append(row)
            
        detailed_df = pd.DataFrame(detailed_rows)
        out_detailed_name = f"predictions_detailed_{info['backbone'].lower().replace(' ', '_')}_{info['task'].lower()}.csv"
        detailed_df.to_csv(os.path.join(reports_dir, out_detailed_name), index=False)
        print(f"  Saved detailed CSV to {reports_dir}/{out_detailed_name}")

    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(reports_dir, "sota_metrics_summary.csv"), index=False)
    print(f"\nSaved summary CSV to {reports_dir}/sota_metrics_summary.csv")

    # Generate Performance Bar Chart
    print("\nGenerating performance comparison bar chart...")
    # Filter for the relevant comparative configurations
    # We want standard vs calibrated for ResNet-18 and PANNs Cnn14
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data manually for plot
    # ResNet ST Std/Cal, ResNet MT Std/Cal, PANNs ST Std/Cal, PANNs MT Std/Cal
    plot_labels = [
        "ResNet-18\nSingle-Task", "ResNet-18\nMulti-Task",
        "PANNs Cnn14\nSingle-Task", "PANNs Cnn14\nMulti-Task"
    ]
    
    std_scores = []
    cal_scores = []
    
    configs = [
        ("hybrid_config_D_resnet_sota_results.json", False),
        ("hybrid_config_D_resnet_sota_multitask_results.json", True),
        ("hybrid_config_D_sota_results.json", False),
        ("hybrid_config_D_sota_multitask_results.json", True)
    ]
    
    for fname, is_mt in configs:
        path = os.path.join(results_dir, fname)
        if os.path.exists(path):
            with open(path, 'r') as f:
                res = json.load(f)
            std_scores.append(res["std"]["score"] * 100)
            cal_scores.append(res["calibrated"]["score"] * 100 if "calibrated" in res else 0.0)
        else:
            std_scores.append(0.0)
            cal_scores.append(0.0)
            
    x = np.arange(len(plot_labels))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, std_scores, width, label='Standard (Argmax)', color='#4A90E2')
    rects2 = ax.bar(x + width/2, cal_scores, width, label='Calibrated (Product)', color='#50E3C2')
    
    ax.set_ylabel('ICBHI Score (%)', fontsize=11, fontweight='bold')
    ax.set_title('SOTA Comparative Performance: ResNet-18 vs PANNs Cnn14', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_labels, fontsize=10)
    ax.set_ylim(0, 60)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend(frameon=True, facecolor='white', edgecolor='none')
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
                            
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "sota_performance_comparison.png"), dpi=300)
    plt.close()
    print(f"Saved performance chart to {reports_dir}/sota_performance_comparison.png")
    
    # Generate Confusion Matrices for ResNet-18 Multi-Task SOTA
    print("\nGenerating confusion matrices for ResNet-18 Multi-Task SOTA...")
    resnet_mt_path = os.path.join(results_dir, "hybrid_config_D_resnet_sota_multitask_results.json")
    if os.path.exists(resnet_mt_path):
        with open(resnet_mt_path, 'r') as f:
            res = json.load(f)
            
        std_preds = res["std"]["preds"]
        cal_preds = res["calibrated"]["preds"] if "calibrated" in res else None
        
        classes = ['Normal', 'Crackle', 'Wheeze', 'Both']
        
        # Plot side-by-side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
        
        # Standard Confusion Matrix
        cm_std = confusion_matrix(cycle_labels_mt, std_preds, normalize='true')
        disp_std = ConfusionMatrixDisplay(confusion_matrix=cm_std, display_labels=classes)
        disp_std.plot(cmap=plt.cm.Blues, ax=ax1, colorbar=False, values_format='.2f')
        ax1.set_title("Standard Predictions (Argmax)", fontsize=11, fontweight='bold')
        
        if cal_preds:
            # Calibrated Confusion Matrix
            cm_cal = confusion_matrix(cycle_labels_mt, cal_preds, normalize='true')
            disp_cal = ConfusionMatrixDisplay(confusion_matrix=cm_cal, display_labels=classes)
            disp_cal.plot(cmap=plt.cm.Blues, ax=ax2, colorbar=False, values_format='.2f')
            ax2.set_title("Calibrated Predictions (Geometric Product)", fontsize=11, fontweight='bold')
            
        plt.suptitle("ResNet-18 SOTA Multi-Task Anomaly Classification Confusion Matrices (Normalized)", fontsize=13, fontweight='bold', y=0.96)
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, "resnet_sota_confusion_matrix.png"), dpi=300)
        plt.close()
        print(f"Saved confusion matrices to {reports_dir}/resnet_sota_confusion_matrix.png")
        
    print("\nAll tasks completed successfully!")

if __name__ == "__main__":
    main()
