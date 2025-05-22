# Tamer Hosny Audio & Lyrics Analysis

This project combines audio and textual deep learning methods to analyze the work of the Egyptian artist **Tamer Hosny**, focusing on two key objectives:

1. **Voice-Based Singer Classification**  
2. **Emotion Analysis via Text and Audio**

---

## Project Summary

- **CNN-Based Voice Classifier**  
  Trained a Convolutional Neural Network (CNN) to determine if the singer of a given song is **Tamer Hosny** or not, using audio spectrograms as input.

- **Emotion Analysis via Lyrics (LLM)**  
  Used a large language model (Qwen3) to classify the **emotional tone** of each song based on its lyrics, with special attention to romantic, patriotic, and introspective themes.

- **Emotion Detection via Audio (LLM)**  
  Applied large language models (LLMs) to transcribed vocals or raw vocal features for **emotion classification** based on vocal delivery, intonation, and expression.

- **Lyrics Analytics and Thematic Trends**  
  Performed a detailed temporal and statistical analysis of 140+ Tamer Hosny songs to explore linguistic patterns, lexical diversity, and thematic evolution over time.

---

## Data Sources

- **Lyrics**: Scraped and verified from Genius API, Wikipedia, and manual transcription.  
- **Audio**: MP3s sourced and stored for model training and evaluation.  
- **Metadata**: Included song title, release year, album, lyricist, and composer.  
- **Custom Arabic Stopwords**: Compiled over 7,000 words to enhance text preprocessing.  

---

##  Tools & Methods

- **Audio Analysis**: CNNs on spectrograms for speaker recognition.  
- **Text NLP**: TF-IDF, PoS tagging, word clouds, and lexical density metrics.  
- **LLMs**: Qwen3 for emotion classification from both lyrics and vocal content.  
- **Visualization**: Word clouds, histograms, temporal charts, PoS-tagged clustering.  

---

## Key Findings

- Romantic and emotional expressions dominate Tamer Hosny’s discography.  
- Lexical diversity increased over time, especially in songs with deeper, more mature themes.  
- Three lyrical clusters emerged: **Romantic/Youthful**, **Nationalistic/Motivational**, and **Poetic/Reflective**.  
- CNN achieved reliable performance in distinguishing Tamer Hosny's voice from others in unseen tracks.  

---

## Future Work

- Enhance audio-based emotion analysis using pretrained speech emotion models.  
- Expand to multi-artist voice and lyric classification.  
- Combine lyrics and audio features in multimodal emotion models.  
- Incorporate music genre classification using PoS + TF-IDF hybrid features.
