import os
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, lfilter
from tqdm import tqdm

# Configuration
DATA_DIR = os.path.join("data", "ICBHI_final_database")
SPLIT_FILE = os.path.join("data", "ICBHI_challenge_train_test.txt")
DIAGNOSIS_FILE = os.path.join("data", "ICBHI_Challenge_diagnosis.txt")
OUTPUT_DIR = "processed_data"

SAMPLE_RATE = 4000  # 4 kHz is sufficient for lung sounds (max 2 kHz)
FILTER_LOW = 100    # Cut off heart sounds (< 100 Hz)
FILTER_HIGH = 1999  # Limit to upper range of lung sounds (must be < Nyquist freq of 2000 Hz)
CYCLE_DURATION = 3.0 # Standardize to 3 seconds

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def load_splits_and_diagnoses():
    # Load Train/Test splits
    splits = {}
    with open(SPLIT_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                splits[parts[0]] = parts[1]
                
    # Load Patient Diagnoses
    diagnoses = {}
    with open(DIAGNOSIS_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                diagnoses[int(parts[0])] = parts[1]
    return splits, diagnoses

def pad_or_truncate(audio, target_length):
    if len(audio) >= target_length:
        return audio[:target_length]
    else:
        return np.pad(audio, (0, target_length - len(audio)), mode='constant')

def main():
    splits, diagnoses = load_splits_and_diagnoses()
    
    # Create target directories
    for split in ['train', 'test']:
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)
        
    # Get all WAV files
    wav_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.wav')]
    print(f"Found {len(wav_files)} raw audio files. Processing...")
    
    cycle_counter = 0
    
    for wav_file in tqdm(wav_files):
        base_name = os.path.splitext(wav_file)[0]
        patient_id = int(base_name.split('_')[0])
        
        # Determine split (default to train if missing)
        split = splits.get(base_name, 'train')
        diagnosis = diagnoses.get(patient_id, 'Unknown')
        
        # Load annotation file
        ann_path = os.path.join(DATA_DIR, base_name + '.txt')
        if not os.path.exists(ann_path):
            continue
            
        annotations = []
        with open(ann_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    # start, end, crackle, wheeze
                    annotations.append((float(parts[0]), float(parts[1]), int(parts[2]), int(parts[3])))
                    
        if not annotations:
            continue
            
        # Load raw audio (resampled to 4kHz)
        audio_path = os.path.join(DATA_DIR, wav_file)
        try:
            audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        except Exception as e:
            print(f"Error loading {wav_file}: {e}")
            continue
            
        # Apply bandpass filter
        filtered_audio = butter_bandpass_filter(audio, FILTER_LOW, FILTER_HIGH, SAMPLE_RATE)
        
        # Segment into individual respiration cycles
        for i, (start, end, crackle, wheeze) in enumerate(annotations):
            # Calculate indices
            start_idx = int(start * SAMPLE_RATE)
            end_idx = int(end * SAMPLE_RATE)
            
            # Extract segment
            segment = filtered_audio[start_idx:end_idx]
            
            # Skip empty segments
            if len(segment) == 0:
                continue
                
            # Standardize length (3.0 seconds)
            target_samples = int(CYCLE_DURATION * SAMPLE_RATE)
            processed_segment = pad_or_truncate(segment, target_samples)
            
            # Normalize segment amplitude
            max_val = np.max(np.abs(processed_segment))
            if max_val > 0:
                processed_segment = processed_segment / max_val
                
            # Determine class label
            # 0: Normal, 1: Crackle, 2: Wheeze, 3: Both
            if crackle == 1 and wheeze == 1:
                class_label = 3
            elif crackle == 1:
                class_label = 1
            elif wheeze == 1:
                class_label = 2
            else:
                class_label = 0
                
            # Filename schema: baseName_cycleIndex_classLabel_diagnosis.wav
            out_filename = f"{base_name}_cycle{i}_class{class_label}_{diagnosis}.wav"
            out_path = os.path.join(OUTPUT_DIR, split, out_filename)
            
            # Save segment
            sf.write(out_path, processed_segment, SAMPLE_RATE)
            cycle_counter += 1
            
    print(f"\nProcessing complete! Segmented and saved {cycle_counter} respiratory cycles.")
    print(f"Data saved to directory: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
