# sudo /Applications/Python\ 3.11/Install\ Certificates.command
# pip install coqui-tts torch torchvision torchaudio nltk

import nltk
import os
from TTS.api import TTS
import torchaudio
import torch

nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

# Read the text file
with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Split text into sentences
sentences = sent_tokenize(text)


# Initialize the model and set it to use CPU
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
tts.to('cpu')
# Directory to save individual audio files
output_dir = 'audio_segments'
os.makedirs(output_dir, exist_ok=True)
# List available speakers

# available_speakers = tts.speakers
# print("Available speakers:", available_speakers)


# Synthesize speech for each sentence
for i, sentence in enumerate(sentences):
    output_path = os.path.join(output_dir, f'segment_{i}.wav')
    tts.tts_to_file(text=sentence, file_path=output_path,speaker='Kumar Dahl',language='en')



# List to hold audio tensors
audio_tensors = []

# Load each audio segment
for i in range(len(sentences)):
    segment_path = os.path.join(output_dir, f'segment_{i}.wav')
    waveform, sample_rate = torchaudio.load(segment_path)
    audio_tensors.append(waveform)

# Concatenate all audio tensors along the time dimension
combined_audio = torch.cat(audio_tensors, dim=1)

# Save the combined audio to a file
torchaudio.save('final_output.wav', combined_audio, sample_rate=sample_rate)



# Remove individual audio segment files
for i in range(len(sentences)):
    segment_path = os.path.join(output_dir, f'segment_{i}.wav')
    os.remove(segment_path)

# Optionally, remove the directory if it's empty
os.rmdir(output_dir)
