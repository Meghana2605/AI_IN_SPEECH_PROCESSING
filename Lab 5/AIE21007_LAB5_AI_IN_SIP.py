#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Load the speech signal
file_path = 'AISP_VOICE_21007.wav'
sample_rate, speech_signal = wavfile.read(file_path)

# Perform FFT to transform to frequency domain
spectral_components = np.fft.fft(speech_signal)

# Plot the amplitude part of the spectral components
plt.figure(figsize=(10, 5))
plt.plot(np.abs(spectral_components))
plt.title('Amplitude of Spectral Components')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.show()

# Inverse transform to time domain signal
time_domain_signal = np.fft.ifft(spectral_components)


# In[14]:


import numpy as np
import matplotlib.pyplot as plt

# Assuming spectral_components is your data with shape (101656, 2)
spectral_components = np.random.randn(101656, 2)

# Assuming you have defined low_freq and high_freq
low_freq = 1000
high_freq = 5000

# Apply cosine window for low pass
cosine_window = np.cos(np.linspace(0, np.pi, len(spectral_components)))
cosine_window = cosine_window[:, np.newaxis]  # Add new axis to match the shape of spectral_components
filtered_spectrum_cosine_low = spectral_components * cosine_window

# Apply Gaussian window for band pass
gaussian_window = np.exp(-0.5 * ((np.arange(len(spectral_components)) - (low_freq + high_freq) / 2) / 1000)**2)
gaussian_window = gaussian_window[:, np.newaxis]  # Add new axis to match the shape of spectral_components
filtered_spectrum_gaussian_band = spectral_components * gaussian_window

# Apply Gaussian window for high pass
filtered_spectrum_gaussian_high = spectral_components * (1 - gaussian_window)

# Inverse transform to time domain signals
filtered_signal_cosine_low = np.fft.ifft(filtered_spectrum_cosine_low)
filtered_signal_gaussian_band = np.fft.ifft(filtered_spectrum_gaussian_band)
filtered_signal_gaussian_high = np.fft.ifft(filtered_spectrum_gaussian_high)

# Print output for cosine window
print("Cosine Window Filtered Signal:")
plt.figure(figsize=(10, 3))
plt.plot(np.real(filtered_signal_cosine_low))
plt.title('Cosine Window Filtered Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

# Print output for Gaussian window (band pass)
print("Gaussian Window Band Pass Filtered Signal:")
plt.figure(figsize=(10, 3))
plt.plot(np.real(filtered_signal_gaussian_band))
plt.title('Gaussian Window Band Pass Filtered Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

# Print output for Gaussian window (high pass)
print("Gaussian Window High Pass Filtered Signal:")
plt.figure(figsize=(10, 3))
plt.plot(np.real(filtered_signal_gaussian_high))
plt.title('Gaussian Window High Pass Filtered Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()


# In[ ]:




