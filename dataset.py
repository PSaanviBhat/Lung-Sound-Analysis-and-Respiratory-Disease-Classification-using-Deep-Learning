import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

class LungSoundDataset(Dataset):
    def __init__(self, metadata_file, split='train', val_ratio=0.2, random_seed=42, transform=None):
        self.split = split
        self.transform = transform
        
        # Load metadata index
        df = pd.read_csv(metadata_file)
        
        # Perform subject-level train/val split
        if split in ['train', 'val']:
            # Filter for official train split
            train_df = df[df['split'] == 'train'].copy()
            unique_patients = sorted(train_df['patient_id'].unique())
            
            # Split patients using seed for reproducibility
            np.random.seed(random_seed)
            np.random.shuffle(unique_patients)
            
            val_size = int(len(unique_patients) * val_ratio)
            val_patients = set(unique_patients[:val_size])
            train_patients = set(unique_patients[val_size:])
            
            if split == 'train':
                self.df = train_df[train_df['patient_id'].isin(train_patients)].reset_index(drop=True)
            else:
                self.df = train_df[train_df['patient_id'].isin(val_patients)].reset_index(drop=True)
        else:
            # Test split
            self.df = df[df['split'] == 'test'].reset_index(drop=True)
            
        print(f"Loaded {len(self.df)} cycles for split: {split} (Unique Patients: {self.df['patient_id'].nunique()})")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row['filepath']
        label = int(row['class_label'])
        
        # Load the pre-computed 3-channel tensor
        tensor = torch.load(file_path)
        
        # Apply data augmentations (only for training)
        if self.split == 'train' and self.transform:
            tensor = self.transform(tensor)
            
        return tensor, label

    def get_labels(self):
        return self.df['class_label'].values

# Data Augmentations (SpecAugment for PyTorch Tensors)
class SpecAugment(object):
    def __init__(self, time_mask_max=15, freq_mask_max=15):
        self.time_mask_max = time_mask_max
        self.freq_mask_max = freq_mask_max

    def __call__(self, tensor):
        # input shape: (3, 128, 128)
        c, h, w = tensor.shape
        
        # Apply Frequency Masking
        f = np.random.randint(0, self.freq_mask_max)
        f0 = np.random.randint(0, h - f)
        tensor[:, f0:f0+f, :] = 0.0
        
        # Apply Time Masking
        t = np.random.randint(0, self.time_mask_max)
        t0 = np.random.randint(0, w - t)
        tensor[:, :, t0:t0+t] = 0.0
        
        return tensor

def get_class_balanced_sampler(dataset):
    labels = dataset.get_labels()
    class_counts = np.bincount(labels)
    class_weights = 1.0 / (class_counts + 1e-8)
    
    # Calculate sample weights based on inverse class frequency
    sample_weights = [class_weights[label] for label in labels]
    sample_weights = torch.DoubleTensor(sample_weights)
    
    # Create WeightedRandomSampler
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

def get_dataloaders(metadata_file="metadata.csv", batch_size=32, num_workers=0):
    # Instantiate training dataset with augmentations
    train_dataset = LungSoundDataset(metadata_file, split='train', transform=SpecAugment())
    val_dataset = LungSoundDataset(metadata_file, split='val')
    test_dataset = LungSoundDataset(metadata_file, split='test')
    
    # Balanced sampler for training to combat class imbalance
    train_sampler = get_class_balanced_sampler(train_dataset)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Validation and test loaders (no sampler, standard shuffle=False)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Quick debug verification run
    print("Testing Dataloader setup...")
    train_l, val_l, test_l = get_dataloaders(batch_size=16)
    
    for batch_x, batch_y in train_l:
        print(f"Batch loaded successfully! Input shape: {batch_x.shape}, Labels shape: {batch_y.shape}")
        # Verify classes are balanced in the batch
        print("Class distributions in this batch:", np.bincount(batch_y.numpy(), minlength=4))
        break
