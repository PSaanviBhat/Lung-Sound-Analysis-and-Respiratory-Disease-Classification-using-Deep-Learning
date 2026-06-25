import os
import numpy as np
import librosa
import pywt
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

# Configuration
INPUT_DIR = "processed_data"
OUTPUT_DIR = "processed_features"
METADATA_FILE = "metadata.csv"

SAMPLE_RATE = 4000
IMAGE_SIZE = 128  # Target height and width for CNN input

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min() + 1e-8)
    X_scaled = X_std * (max - min) + min
    return X_scaled

def resize_tensor(data_np, target_size=(IMAGE_SIZE, IMAGE_SIZE)):
    # Convert numpy array to torch tensor and add batch/channel dimensions
    tensor = torch.tensor(data_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    # Resize using bilinear interpolation
    resized = F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)
    # Remove batch/channel dimensions and convert back to numpy
    return resized.squeeze(0).squeeze(0).numpy()

def extract_features(audio):
    # 1. Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=IMAGE_SIZE, n_fft=512, hop_length=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_scaled = scale_minmax(mel_spec_db)
    mel_spec_resized = resize_tensor(mel_spec_scaled)

    # 2. Constant-Q Transform (CQT)
    # Note: Use a fallback to Mel Spectrogram if the audio segment is too short for CQT
    try:
        cqt = librosa.cqt(y=audio, sr=SAMPLE_RATE, hop_length=128, n_bins=IMAGE_SIZE, bins_per_octave=24, fmin=librosa.note_to_hz('C1'))
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        cqt_scaled = scale_minmax(cqt_db)
        cqt_resized = resize_tensor(cqt_scaled)
    except Exception:
        # Fallback to copy Mel Spectrogram if CQT fails on a specific edge case
        cqt_resized = mel_spec_resized

    # 3. Continuous Wavelet Transform (CWT) Scalogram
    # Using 'mexh' (Mexican Hat/Ricker wavelet) which is ideal for transient/impulsive crackles
    scales = np.arange(1, IMAGE_SIZE + 1)
    coefs, freqs = pywt.cwt(audio, scales, 'mexh')
    cwt_scaled = scale_minmax(np.abs(coefs))
    cwt_resized = resize_tensor(cwt_scaled)

    # Stack into a 3-channel representation (3, 128, 128)
    stacked = np.stack([mel_spec_resized, cqt_resized, cwt_resized], axis=0)
    return stacked

def main():
    metadata_records = []
    
    # Create target directories
    for split in ['train', 'test']:
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)
        
    for split in ['train', 'test']:
        split_dir = os.path.join(INPUT_DIR, split)
        if not os.path.exists(split_dir):
            continue
            
        files = [f for f in os.listdir(split_dir) if f.endswith('.wav')]
        print(f"\nExtracting features for {split} split ({len(files)} files)...")
        
        for file in tqdm(files):
            # Filename schema: baseName_cycleIndex_classLabel_diagnosis.wav
            base_name = os.path.splitext(file)[0]
            parts = base_name.split('_')
            
            patient_id = int(parts[0])
            cycle_idx = parts[-3]
            class_label = int(parts[-2].replace('class', ''))
            diagnosis = parts[-1]
            
            audio_path = os.path.join(split_dir, file)
            try:
                # Load segmented 3s cycle wav file
                audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
                
            # Extract 3-channel tensor
            feature_tensor = extract_features(audio)
            
            # Save as PyTorch tensor
            out_filename = f"{base_name}.pt"
            out_path = os.path.join(OUTPUT_DIR, split, out_filename)
            torch.save(torch.tensor(feature_tensor, dtype=torch.float32), out_path)
            
            # Append metadata records
            metadata_records.append({
                'filepath': os.path.join(OUTPUT_DIR, split, out_filename),
                'patient_id': patient_id,
                'cycle_idx': cycle_idx,
                'class_label': class_label,
                'diagnosis': diagnosis,
                'split': split
            })
            
    # Save metadata index to CSV
    df = pd.DataFrame(metadata_records)
    df.to_csv(METADATA_FILE, index=False)
    print(f"\nFeature extraction complete! Saved {len(df)} feature tensors.")
    print(f"Metadata index file written to: {os.path.abspath(METADATA_FILE)}")

if __name__ == "__main__":
    main()
