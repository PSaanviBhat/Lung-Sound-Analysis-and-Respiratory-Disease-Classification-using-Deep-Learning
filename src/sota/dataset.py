import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

class SOTAAugmentation(object):
    def __init__(self, time_shift_max=12, freq_shift_max=2, noise_level_max=0.03, time_mask_max=15, freq_mask_max=15):
        self.time_shift_max = time_shift_max
        self.freq_shift_max = freq_shift_max
        self.noise_level_max = noise_level_max
        self.time_mask_max = time_mask_max
        self.freq_mask_max = freq_mask_max

    def __call__(self, tensor):
        # Input shape: (C, H, W) e.g., (3, 128, 128)
        
        # 1. Random Time Shifting (horizontal circular shift)
        if self.time_shift_max > 0:
            shift = np.random.randint(-self.time_shift_max, self.time_shift_max + 1)
            if shift != 0:
                tensor = torch.roll(tensor, shifts=shift, dims=2)
                
        # 2. Random Frequency Shifting / Pitch Shifting (vertical circular shift)
        if self.freq_shift_max > 0:
            shift = np.random.randint(-self.freq_shift_max, self.freq_shift_max + 1)
            if shift != 0:
                tensor = torch.roll(tensor, shifts=shift, dims=1)
                
        # 3. Gaussian Noise Injection
        if self.noise_level_max > 0:
            noise_level = np.random.uniform(0.0, self.noise_level_max)
            noise = torch.randn_like(tensor) * noise_level
            tensor = tensor + noise
            # Clamp to maintain [0, 1] range after noise injection
            tensor = torch.clamp(tensor, 0.0, 1.0)
            
        # 4. SpecAugment (Frequency & Time Masking)
        c, h, w = tensor.shape
        if self.freq_mask_max > 0:
            f = np.random.randint(0, self.freq_mask_max)
            if f > 0:
                f0 = np.random.randint(0, h - f)
                tensor[:, f0:f0+f, :] = 0.0
                
        if self.time_mask_max > 0:
            t = np.random.randint(0, self.time_mask_max)
            if t > 0:
                t0 = np.random.randint(0, w - t)
                tensor[:, :, t0:t0+t] = 0.0
                
        return tensor

class LungSoundDatasetSOTA(Dataset):
    def __init__(self, metadata_file, split='train', val_ratio=0.2, random_seed=42, transform=None, channels_to_use=None, multitask=False):
        self.split = split
        self.transform = transform
        self.channels_to_use = channels_to_use
        self.multitask = multitask
        
        # Load metadata index
        df = pd.read_csv(metadata_file)
        
        if self.multitask:
            # Filter for COPD, URTI, Healthy
            df = df[df['diagnosis'].isin(['COPD', 'URTI', 'Healthy'])].reset_index(drop=True)
            self.pathology_map = {'COPD': 0, 'URTI': 1, 'Healthy': 2}
            df['pathology_label'] = df['diagnosis'].map(self.pathology_map)
            
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
        
        # Squeeze/slice active channels if defined
        if self.channels_to_use is not None:
            tensor = tensor[self.channels_to_use, :, :]
            
        # Apply SOTA data augmentations (only for training)
        if self.split == 'train' and self.transform:
            tensor = self.transform(tensor)
            
        if self.multitask:
            pathology_label = int(row['pathology_label'])
            return tensor, label, pathology_label
            
        return tensor, label

    def get_labels(self):
        return self.df['class_label'].values

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

def get_dataloaders(metadata_file="metadata.csv", batch_size=32, num_workers=0, channels_to_use=None, use_augmentations=True, multitask=False):
    # Instantiate SOTA training dataset with augmentations
    transform = SOTAAugmentation() if use_augmentations else None
    
    train_dataset = LungSoundDatasetSOTA(metadata_file, split='train', transform=transform, channels_to_use=channels_to_use, multitask=multitask)
    val_dataset = LungSoundDatasetSOTA(metadata_file, split='val', channels_to_use=channels_to_use, multitask=multitask)
    test_dataset = LungSoundDatasetSOTA(metadata_file, split='test', channels_to_use=channels_to_use, multitask=multitask)
    
    # Balanced sampler for training to combat class imbalance
    train_sampler = get_class_balanced_sampler(train_dataset)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler, 
        num_workers=num_workers,
        pin_memory=True
    )
    
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
    print("Testing SOTA Dataloader setup...")
    train_l, val_l, test_l = get_dataloaders(batch_size=16)
    
    for batch_x, batch_y in train_l:
        print(f"Batch loaded successfully! Input shape: {batch_x.shape}, Labels shape: {batch_y.shape}")
        break
