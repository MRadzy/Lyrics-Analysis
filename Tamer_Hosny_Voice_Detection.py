import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tempfile
import av
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import joblib
import time
import torchaudio
import torchaudio.transforms as T
import random

FEATURE_CACHE = {}
CACHE_FILE = 'feature_cache.joblib'

def load_cache():
    global FEATURE_CACHE
    if os.path.exists(CACHE_FILE):
        try:
            FEATURE_CACHE = joblib.load(CACHE_FILE)
            print(f"Loaded {len(FEATURE_CACHE)} cached features")
        except:
            FEATURE_CACHE = {}
    else:
        FEATURE_CACHE = {}

def save_cache():
    joblib.dump(FEATURE_CACHE, CACHE_FILE)
    print(f"Saved {len(FEATURE_CACHE)} features to cache")

def extract_audio(filepath, duration=30, sr=16000):
    """Extract audio from video files and convert to the required format"""
    if filepath.endswith(('.mp4', '.webm')):
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            container = av.open(filepath)
            audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
            
            if audio_stream is None:
                print(f"No audio stream found in {filepath}")
                return None
                
            output = av.open(temp_path, 'w')
            output_stream = output.add_stream('pcm_s16le', rate=sr)
            
            for frame in container.decode(audio_stream):
                frame.pts = None
                packet = output_stream.encode(frame)
                output.mux(packet)
                
            packet = output_stream.encode(None)
            output.mux(packet)
            output.close()
            container.close()
            
            waveform, sample_rate = torchaudio.load(temp_path)
            os.remove(temp_path)
            
            if sample_rate != sr:
                resampler = torchaudio.transforms.Resample(sample_rate, sr)
                waveform = resampler(waveform)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            target_length = sr * duration
            if waveform.shape[1] > target_length:
                waveform = waveform[:, :target_length]
            else:
                padding = target_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            return waveform
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return None
    else:
        try:
            waveform, sample_rate = torchaudio.load(filepath)
            
            if sample_rate != sr:
                resampler = torchaudio.transforms.Resample(sample_rate, sr)
                waveform = resampler(waveform)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            target_length = sr * duration
            if waveform.shape[1] > target_length:
                waveform = waveform[:, :target_length]
            else:
                padding = target_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            return waveform
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return None

class AudioAugmentation:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
    def add_noise(self, waveform, noise_level=0.005):
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise
    
    def time_stretch(self, waveform, rate_range=(0.9, 1.1)):
        rate = random.uniform(*rate_range)
        effects = [["speed", str(rate)], ["rate", str(self.sample_rate)]]
        return torchaudio.sox_effects.apply_effects_tensor(waveform, self.sample_rate, effects)[0]
    
    def pitch_shift(self, waveform, pitch_range=(-2, 2)):
        pitch = random.uniform(*pitch_range)
        effects = [["pitch", str(pitch)], ["rate", str(self.sample_rate)]]
        return torchaudio.sox_effects.apply_effects_tensor(waveform, self.sample_rate, effects)[0]
    
    def frequency_mask(self, mel_spec, freq_mask_param=30):
        return T.FrequencyMasking(freq_mask_param)(mel_spec)
    
    def time_mask(self, mel_spec, time_mask_param=100):
        return T.TimeMasking(time_mask_param)(mel_spec)
    
    def apply_random_augmentation(self, waveform, mel_spec=None):
        augmentations = [
            (self.add_noise, 0.7),  
            (self.time_stretch, 0.5),  
            (self.pitch_shift, 0.5),  
        ]
        for aug, prob in augmentations:
            if random.random() < prob:
                waveform = aug(waveform)
        
        if mel_spec is not None:
            if random.random() < 0.5:  # 50% chance for each
                mel_spec = self.frequency_mask(mel_spec)
            if random.random() < 0.5:  # 50% chance for each
                mel_spec = self.time_mask(mel_spec)
        
        return waveform, mel_spec

class SongDataset(Dataset):
    def __init__(self, filepaths, labels, augment=False):
        self.filepaths = filepaths
        self.labels = labels
        self.augment = augment
        self.mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            f_min=20,
            f_max=8000
        )
        self.augmentor = AudioAugmentation() if augment else None
    
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        try:
            # Check cache first
            if self.filepaths[idx] in FEATURE_CACHE:
                features = FEATURE_CACHE[self.filepaths[idx]]
            else:
                # Extract audio
                waveform = extract_audio(self.filepaths[idx])
                if waveform is None:
                    raise ValueError(f"Could not extract audio from {self.filepaths[idx]}")
                
                # Apply augmentation if enabled
                if self.augment and self.augmentor:
                    waveform, features = self.augmentor.apply_random_augmentation(waveform, self.mel_transform(waveform))
                
                # Cache the features
                FEATURE_CACHE[self.filepaths[idx]] = features
            
            label = torch.tensor(self.labels[idx]).float()
            return features, label
            
        except Exception as e:
            print(f"Error processing item {idx} ({self.filepaths[idx]}): {e}")
            # Return dummy features with correct shape
            dummy_features = torch.zeros((128, 938))  # Mel spectrogram shape
            return dummy_features, torch.tensor(self.labels[idx]).float()

class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=1):
        super(ImprovedCNN, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2)
        
        # Adaptive pooling to get fixed size output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Softmax(dim=1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Add channel dimension if not present
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        # Feature extraction
        x = self.conv1(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.pool(x)
        
        x = self.conv4(x)
        x = self.pool(x)
        
        x = self.conv5(x)
        x = self.pool(x)
        
        # Adaptive pooling to get fixed size output
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 512)
        
        # Apply attention
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # Classify
        x = self.classifier(x)
        return x.squeeze(-1)

def visualize_data_distribution(train_labels, test_labels):
    plt.figure(figsize=(12, 5))
    
    # Train set distribution
    plt.subplot(1, 2, 1)
    train_counts = Counter(train_labels)
    labels = ['Random Songs', 'Tamer Hosny Songs']
    values = [train_counts[0], train_counts[1]]
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'])
    plt.title('Training Set Distribution')
    
    # Test set distribution
    plt.subplot(1, 2, 2)
    test_counts = Counter(test_labels)
    values = [test_counts[0], test_counts[1]]
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'])
    plt.title('Test Set Distribution')
    
    plt.tight_layout()
    plt.savefig('data_distribution.png')
    plt.close()
    print("Data distribution visualization saved as 'data_distribution.png'")

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                  display_labels=['Random Songs', 'Tamer Hosny Songs'])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved as 'confusion_matrix.png'")

def load_data_and_labels():
    random_folder = 'random_songs'
    tamer_folder = 'tamer_hosny_songs'
    
    # Include both .mp4 and .webm files
    random_files = [os.path.join(random_folder, f) for f in os.listdir(random_folder) 
                   if f.endswith(('.mp4', '.webm'))]
    tamer_files = [os.path.join(tamer_folder, f) for f in os.listdir(tamer_folder) 
                  if f.endswith(('.mp4', '.webm'))]
    
    files = random_files + tamer_files
    labels = [0]*len(random_files) + [1]*len(tamer_files)
    return files, labels

def collate_fn(batch):
    batch = [(features, label) for features, label in batch if features is not None]
    if not batch:
        return torch.zeros((0, 1, 128, 938)), torch.zeros((0, 1))
        
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    features = torch.stack(features)
    labels = torch.stack(labels)
    
    return features, labels

def plot_metrics_history(metrics_history):
    """Plot training metrics over time"""
    plt.figure(figsize=(15, 10))
    
    # Plot accuracy
    plt.subplot(2, 2, 1)
    plt.plot(metrics_history['train_accuracy'], label='Train')
    plt.plot(metrics_history['val_accuracy'], label='Validation')
    plt.title('Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Plot F1 score
    plt.subplot(2, 2, 2)
    plt.plot(metrics_history['train_f1'], label='Train')
    plt.plot(metrics_history['val_f1'], label='Validation')
    plt.title('F1 Score Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    # Plot precision
    plt.subplot(2, 2, 3)
    plt.plot(metrics_history['train_precision'], label='Train')
    plt.plot(metrics_history['val_precision'], label='Validation')
    plt.title('Precision Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    
    # Plot recall
    plt.subplot(2, 2, 4)
    plt.plot(metrics_history['train_recall'], label='Train')
    plt.plot(metrics_history['val_recall'], label='Validation')
    plt.title('Recall Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()
    print("Training metrics visualization saved as 'training_metrics.png'")

def train():
    start_time = time.time()
    
    # Load feature cache
    load_cache()
    
    files, labels = load_data_and_labels()
    print(f"Total files: {len(files)}, Positive (Tamer) examples: {sum(labels)}, Negative examples: {len(labels) - sum(labels)}")
    
    # Split into train+test and validation sets
    train_test_files, val_files, train_test_labels, val_labels = train_test_split(
        files, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Calculate class weights
    n_samples = len(train_test_labels)
    n_positive = sum(train_test_labels)
    n_negative = n_samples - n_positive
    
    # Calculate positive weight for BCEWithLogitsLoss
    pos_weight = torch.tensor([n_negative / n_positive])
    print(f"Positive weight: {pos_weight}")
    
    # Visualize data distribution
    visualize_data_distribution(train_test_labels, val_labels)
    
    # Create datasets with augmentation for training
    train_dataset = SongDataset(train_test_files, train_test_labels, augment=True)
    val_dataset = SongDataset(val_files, val_labels, augment=False)
    
    # Use larger batch size for faster training
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=collate_fn, num_workers=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize classifier with improved architecture
    classifier = ImprovedCNN().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = optim.AdamW(classifier.parameters(), lr=0.0003, weight_decay=0.01)
    
    # Learning rate scheduler with longer warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=130,
        steps_per_epoch=len(train_loader),
        pct_start=0.4,  # Longer warmup
        div_factor=10,
        final_div_factor=100
    )
    
    epochs = 130
    best_f1 = 0
    best_model_state = None
    best_metrics = None
    
    # Initialize metrics history
    metrics_history = {
        'train_accuracy': [], 'val_accuracy': [],
        'train_f1': [], 'val_f1': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': []
    }
    
    for epoch in range(epochs):
        epoch_start = time.time()
        classifier.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_train_preds = []
        all_train_labels = []
        
        for features, labels in train_loader:
            if features.shape[0] == 0:
                continue
                
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            predicted = (outputs > 0).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Collect predictions and labels for metrics
            all_train_preds.extend(predicted.cpu().numpy().tolist())
            all_train_labels.extend(labels.cpu().numpy().tolist())
            
            # Print predictions distribution
            if total % 100 == 0:
                pred_positive = predicted.sum().item()
                pred_negative = total - pred_positive
                print(f"Predictions so far - Positive: {pred_positive}, Negative: {pred_negative}")
        
        # Calculate training metrics
        train_accuracy = 100 * correct / (total + 1e-8)
        train_f1 = f1_score(all_train_labels, all_train_preds)
        train_precision = precision_score(all_train_labels, all_train_preds)
        train_recall = recall_score(all_train_labels, all_train_preds)
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/(len(train_loader) + 1e-8):.4f}")
        print(f"Train - Accuracy: {train_accuracy:.2f}%, F1: {train_f1:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
        print(f"Time: {epoch_time:.2f}s")
        
        # Evaluate on validation set
        classifier.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                if features.shape[0] == 0:
                    continue
                    
                features, labels = features.to(device), labels.to(device)
                outputs = classifier(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                predicted = (outputs > 0).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Collect predictions and labels for metrics
                all_val_preds.extend(predicted.cpu().numpy().tolist())
                all_val_labels.extend(labels.cpu().numpy().tolist())
        
        # Calculate validation metrics
        val_accuracy = 100 * correct / (total + 1e-8)
        val_f1 = f1_score(all_val_labels, all_val_preds)
        val_precision = precision_score(all_val_labels, all_val_preds)
        val_recall = recall_score(all_val_labels, all_val_preds)
        
        print(f"Validation - Loss: {val_loss/(len(val_loader) + 1e-8):.4f}")
        print(f"Validation - Accuracy: {val_accuracy:.2f}%, F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
        
        # Update metrics history
        metrics_history['train_accuracy'].append(train_accuracy)
        metrics_history['val_accuracy'].append(val_accuracy)
        metrics_history['train_f1'].append(train_f1)
        metrics_history['val_f1'].append(val_f1)
        metrics_history['train_precision'].append(train_precision)
        metrics_history['val_precision'].append(val_precision)
        metrics_history['train_recall'].append(train_recall)
        metrics_history['val_recall'].append(val_recall)
        
        # Save best model based on F1 score
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = classifier.state_dict()
            best_metrics = {
                'epoch': epoch,
                'f1_score': val_f1,
                'accuracy': val_accuracy,
                'precision': val_precision,
                'recall': val_recall
            }
            print(f"New best model with F1 score: {best_f1:.4f}")
            
            # Save the model immediately when we find a better one
            model_save_path = 'best_model.pth'
            torch.save({
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'f1_score': val_f1,
                'accuracy': val_accuracy,
                'precision': val_precision,
                'recall': val_recall,
                'metrics_history': metrics_history
            }, model_save_path)
            print(f"Best model saved to {model_save_path}")
    
    # Plot metrics history
    plot_metrics_history(metrics_history)
    
    # Load best model for final evaluation
    if best_model_state is not None:
        classifier.load_state_dict(best_model_state)
        print("\nLoaded best model from epoch", best_metrics['epoch'])
        print("Best model metrics:")
        print(f"F1 Score: {best_metrics['f1_score']:.4f}")
        print(f"Accuracy: {best_metrics['accuracy']:.2f}%")
        print(f"Precision: {best_metrics['precision']:.4f}")
        print(f"Recall: {best_metrics['recall']:.4f}")
    
    # Save feature cache
    save_cache()
    
    # Final evaluation with best model
    classifier.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in val_loader:
            if features.shape[0] == 0:
                continue
                
            features, labels = features.to(device), labels.to(device)
            outputs = classifier(features)
            predicted = (outputs > 0).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Collect predictions and labels for confusion matrix
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    
    test_accuracy = 100 * correct / (total + 1e-8)
    test_f1 = f1_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds)
    test_recall = recall_score(all_labels, all_preds)
    
    total_time = time.time() - start_time
    print("\nFinal Validation Results (using best model):")
    print(f"Accuracy: {test_accuracy:.2f}%")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"Total training time: {total_time:.2f} seconds")
    
    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds)

if __name__ == "__main__":
    train()