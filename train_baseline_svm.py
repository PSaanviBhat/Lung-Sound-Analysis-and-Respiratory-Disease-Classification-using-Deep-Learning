import os
import torch
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tqdm import tqdm

# Configuration
METADATA_FILE = "metadata.csv"

def extract_statistical_features(filepath):
    # Load tensor shape (3, 128, 128)
    tensor = torch.load(filepath).numpy()
    
    features = []
    for channel in range(3):
        channel_data = tensor[channel]  # Shape (128, 128)
        
        # Calculate mean and standard deviation along the time dimension (axis 1)
        mean_feats = np.mean(channel_data, axis=1)  # 128 features
        std_feats = np.std(channel_data, axis=1)    # 128 features
        
        features.extend(mean_feats)
        features.extend(std_feats)
        
    return np.array(features)  # Total: 3 * (128 + 128) = 768 features

def load_data_split(df, split='train'):
    X = []
    y = []
    
    # Filter for active split
    split_df = df[df['split'] == split]
    print(f"Loading features for {split} split ({len(split_df)} files)...")
    
    for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
        filepath = row['filepath']
        label = int(row['class_label'])
        
        # Extract features
        features = extract_statistical_features(filepath)
        X.append(features)
        y.append(label)
        
    return np.array(X), np.array(y)

def main():
    if not os.path.exists(METADATA_FILE):
        print(f"Error: {METADATA_FILE} not found! Please run Phase 3 first.")
        return
        
    # Load metadata index
    df = pd.read_csv(METADATA_FILE)
    
    # 1. Load Train and Test Data splits
    X_train, y_train = load_data_split(df, 'train')
    X_test, y_test = load_data_split(df, 'test')
    
    # 2. Normalize features
    print("Normalizing features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Train Support Vector Machine (RBF Kernel)
    print("Training SVM classifier (RBF kernel, balanced class weights)...")
    # class_weight='balanced' helps SVM handle the remaining class imbalance
    svm = SVC(kernel='rbf', C=1.0, class_weight='balanced', random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    # 4. Predict and Evaluate
    print("Evaluating SVM on test split...")
    preds = svm.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    
    print(f"\nFinal SVM Test Results | Accuracy: {acc*100:.2f}% | Macro F1-Score: {f1:.4f}")
    
    print("\nDetailed SVM Test Classification Report:")
    target_names = ['Normal', 'Crackle', 'Wheeze', 'Both']
    print(classification_report(y_test, preds, target_names=target_names, zero_division=0))

if __name__ == "__main__":
    main()
