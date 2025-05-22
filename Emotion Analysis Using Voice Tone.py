import os
import torch
import torchaudio
import tempfile
import av
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
import requests
import numpy as np

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

def analyze_emotion_with_ollama(audio_path):
    """Analyze the emotion of a song using Ollama's Qwen model"""
    prompt = """Analyze the emotion of this Arabic song. Choose from these emotions:
    - Love (حب)
    - Sadness (حزن)
    - Happiness (سعادة)
    - Longing (شوق)
    - Passion (شغف)
    - Hope (أمل)
    - Nostalgia (حنين)
    - Pride (فخر)
    - Anger (غضب)
    - Peace (سلام)
    
    Respond with only the emotion name in English and Arabic, separated by a comma."""
    
    song_name = os.path.basename(audio_path)
    
    full_prompt = f"Song: {song_name}\n{prompt}"
    
    response = requests.post('http://localhost:11434/api/generate',
                           json={
                               "model": "qwen3:4b",
                               "prompt": full_prompt,
                               "stream": False
                           })
    
    if response.status_code == 200:
        result = response.json()
        try:
            emotion = result['response'].strip()
            return emotion
        except:
            return "Unknown"
    else:
        print(f"Error calling Ollama API: {response.status_code}")
        return "Unknown"

def plot_emotion_distribution(emotions):
    """Plot the distribution of emotions"""
    emotion_counts = Counter(emotions)
    
    plt.figure(figsize=(12, 6))
    
    bars = plt.bar(emotion_counts.keys(), emotion_counts.values())
    
    plt.title('Distribution of Emotions in Tamer Hosny Songs', fontsize=14)
    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('emotion_distribution.png')
    plt.close()
    print("Emotion distribution plot saved as 'emotion_distribution.png'")

def main():
    tamer_folder = 'tamer_hosny_songs'
    tamer_files = [os.path.join(tamer_folder, f) for f in os.listdir(tamer_folder) 
                  if f.endswith(('.mp4', '.webm'))]
    
    print(f"Found {len(tamer_files)} Tamer Hosny songs")
    
    emotions = []
    results = {}
    
    for file_path in tamer_files:
        print(f"\nAnalyzing: {os.path.basename(file_path)}")
        emotion = analyze_emotion_with_ollama(file_path)
        emotions.append(emotion)
        results[os.path.basename(file_path)] = emotion
        print(f"Detected emotion: {emotion}")
    
    with open('emotion_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print("\nResults saved to 'emotion_results.json'")
    
    plot_emotion_distribution(emotions)

if __name__ == "__main__":
    main()