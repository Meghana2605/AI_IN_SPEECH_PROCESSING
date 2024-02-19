#!/usr/bin/env python
# coding: utf-8

# # 1. Use librosa.effects.trim()to remove the silence parts of speech from beginning and end of your recorded signal.Listen to the new signal and perceptually compare the audio with original.

# In[25]:


# import librosa
import librosa.display
import matplotlib.pyplot as plt
import sounddevice as sd

# Load the original audio signal
file_path = 'AISP_VOICE_21007.wav'
y, sr = librosa.load(file_path)

# Trim the silence from the beginning and end
y_trimmed, index = librosa.effects.trim(y)

# Create a time array for plotting
time = np.linspace(0, len(y) / sr, len(y))
time_trimmed = np.linspace(0, len(y_trimmed) / sr, len(y_trimmed))

# Plot both the original and trimmed audio signals
plt.figure(figsize=(12, 6))

# Plot the original audio signal
plt.plot(time, y, label='Original Audio Signal', alpha=0.7)

# Plot the trimmed audio signal
plt.plot(time_trimmed, y_trimmed, label='Trimmed Audio Signal', alpha=0.7)

plt.title('Original and Trimmed Audio Signals')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Play original audio
print("Playing original audio...")
sd.play(y, sr)
sd.wait()

# Play trimmed audio
print("Playing trimmed audio...")
sd.play(y_trimmed, sr)
sd.wait()


# # 2.Use librosa.effects.split()to splitthe recorded speech with detected silences.Play around with the top_db parameter and see the effects of split. Listen to the generated signals and observe the split quality.

# In[26]:


import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt

# Load the audio file
file_path = "AISP_VOICE_21007.wav"
audio, sr = librosa.load(file_path, sr=None)

# Display the original audio waveform
plt.figure(figsize=(10, 4))
librosa.display.waveshow(audio, sr=sr)
plt.title('Original Audio Waveform')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

# Split the audio using librosa.effects.split() with different top_db values
top_db_values = [10, 20, 100]  # Experiment with different values
split_audios = [librosa.effects.split(audio, top_db=top_db) for top_db in top_db_values]

# Display and listen to the generated signals for each top_db value
for i, (split_audio, top_db) in enumerate(zip(split_audios, top_db_values)):
    plt.figure(figsize=(10, 4))
    plt.title(f'Split Audio (top_db={top_db})')
    for start, end in split_audio:
        plt.axvline(x=start/sr, color='r', linestyle='--')
        plt.axvline(x=end/sr, color='r', linestyle='--')
    librosa.display.waveshow(audio, sr=sr, alpha=0.8)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

    # Play the split audio
    print(f"Split Audio (top_db={top_db}):")
    ipd.display(ipd.Audio(audio[start:end], rate=sr))


# In[27]:


import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio

# Load the audio file
file_path = "AISP_VOICE_21007.wav"
signal, sr = librosa.load(file_path, sr=None)

# Perform speech splitting with different top_db values
top_db_values = [18,20, 40, 60,80]  # Adjust these values as desired
split_signals = []

for top_db in top_db_values:
    split_signal = librosa.effects.split(signal, top_db=top_db)
    split_signals.append(split_signal)

# Plot the original and split audio signals for each top_db value
plt.figure(figsize=(12, 8))
plt.subplot(len(top_db_values) + 1, 1, 1)
librosa.display.waveshow(signal, sr=sr)
plt.title('Original Signal')

for i, split_signal in enumerate(split_signals):
    plt.subplot(len(top_db_values) + 1, 1, i + 2)
    split_signal_plot = np.zeros_like(signal)
    for interval in split_signal:
        split_signal_plot[interval[0]:interval[1]] = signal[interval[0]:interval[1]]
    librosa.display.waveshow(split_signal_plot, sr=sr)
    plt.title(f'Split Signal (top_db={top_db_values[i]})')

    # Listen to the split signal
    split_audio = np.concatenate([signal[interval[0]:interval[1]] for interval in split_signal])
    display(Audio(data=split_audio, rate=sr))

plt.tight_layout()
plt.show()


# # 3.

# Lower top_db values (e.g., 18, 20) result in more aggressive splitting, capturing quieter segments as separate splits. 
# This may lead to more fragmented speech with smaller isolated segments.
# Moderate top_db values (e.g., 40) offer a balance between granularity and cohesiveness, capturing moderately quiet segments without excessive fragmentation.
# Higher top_db values (e.g., 60) lead to less aggressive splitting, capturing only relatively loud segments. 
# This may result in fewer, but longer, speech segments.
# The choice of top_db depends on the specific characteristics of your audio data and the desired level of granularity in the split signals

# In[ ]:




