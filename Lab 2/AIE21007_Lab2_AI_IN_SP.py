#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa
import librosa.display


# In[2]:


filename='AISP_VOICE_21007.wav'
y, sr = librosa.load(filename)


# In[3]:


import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load the audio file
file_path = 'AISP_VOICE_21007.wav'
signal, sample_rate = librosa.load(file_path, sr=None)



# Compute the finite difference (first derivative)
delta_t = 1 / sample_rate
first_derivative = np.diff(signal) / delta_t

# Plot the original signal and its first derivative
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
librosa.display.waveshow(signal, sr=sample_rate)
plt.title('Original Speech Signal')

plt.subplot(2, 1, 2)
librosa.display.waveshow(first_derivative, sr=sample_rate)
plt.title('First Derivative of Speech Signal')

plt.tight_layout()
plt.show()


# In[6]:


# Assuming you have already computed the 'time_intervals' array
# Replace 'threshold_speech' and 'threshold_silence' with your actual threshold values

# Set your chosen thresholds (adjust these values as needed)
threshold_speech = 100  # Example: Minimum duration for speech (in samples)
threshold_silence = 50  # Example: Maximum duration for silence (in samples)

# Assuming you have already computed the 'first_derivative' signal
# Replace 'threshold_speech' and 'threshold_silence' with your actual threshold values

# Detect zero crossings
zero_crossings = np.where(np.diff(np.sign(first_derivative)))[0]

# Calculate time intervals between consecutive zero crossings
time_intervals = np.diff(zero_crossings)

# Set your chosen thresholds (adjust these values as needed)
threshold_speech = 100  # Example: Minimum duration for speech (in samples)
threshold_silence = 50  # Example: Maximum duration for silence (in samples)

# Separate speech and silence regions
speech_intervals = time_intervals[time_intervals > threshold_speech]
silence_intervals = time_intervals[time_intervals <= threshold_silence]

# Compute average lengths
average_speech_length = np.mean(speech_intervals)
average_silence_length = np.mean(silence_intervals)

print(f"Average speech length: {average_speech_length:.2f} samples")
print(f"Average silence length: {average_silence_length:.2f} samples")


# In[7]:


import wave

def get_audio_duration(filename):
    with wave.open(filename, 'rb') as audio_file:
        frames = audio_file.getnframes()
        frame_rate = audio_file.getframerate()
        duration = frames / frame_rate
    return duration

def estimate_words(duration, average_wpm=150):
    # Assuming an average speaking rate of 150 words per minute (wpm)
    return int(duration * average_wpm / 60)

# Replace 'LAB_2_5-WORDS.wav' with the actual path to your audio file
audio_file_path = 'AIE21007_LAB_2.wav'
speech_duration = get_audio_duration(audio_file_path)
estimated_word_count = estimate_words(speech_duration)

print(f"Estimated word count in the speech: {estimated_word_count} words")


# In[8]:


import librosa
import numpy as np

def get_pitch(audio_file_path):
    # Load the audio file
    y, sr = librosa.load(audio_file_path)

    # Compute the pitch using Harmonic-Percussive Source Separation
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    # Average pitch across time frames
    avg_pitch = np.mean(pitches)

    return avg_pitch

# Replace with actual file paths
question_pitch = get_pitch('question.wav')
statement_pitch = get_pitch('statement.wav')

# Compare pitch characteristics
print("Pitch analysis results:")
print(f"Question average pitch: {question_pitch:.2f} Hz")
print(f"Statement average pitch: {statement_pitch:.2f} Hz")

# You can also visualize the pitch contours for both sentences if needed.


# In[ ]:




