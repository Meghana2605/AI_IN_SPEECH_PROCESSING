#!/usr/bin/env python
# coding: utf-8

# In[4]:


import librosa
import librosa.display


# In[42]:


filename='AISP_VOICE_21007.wav'
y, sr = librosa.load(filename)


# # A1.Load the recorded speech file into your python workspace. Once loaded, plot the graph for the speech signal.You may use the below code from librosa asa reference.

# In[43]:


librosa.display.waveshow(y)


# In[ ]:





# In[45]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from glob import glob

import librosa
import librosa.display
import IPython.display as ipd

from itertools import cycle

sns.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

# Trimming leading/lagging silence
y_trimmed, _ = librosa.effects.trim(y, top_db=20)
pd.Series(y_trimmed).plot(figsize=(10, 5),
                  lw=1,
                  title='Raw Audio Trimmed Example',
                 color=color_pal[1])
plt.show()


# # A3. Take a small segment of the signal and play it. 

# In[46]:


start_time = 0.2
end_time = 1.0

# Convert time to samples
start_sample = int(start_time * sr)
end_sample = int(end_time * sr)

# Segment the audio based on time
y_segmented = y[start_sample:end_sample]

# Plot the segmented audio
plt.figure(figsize=(7, 3))
plt.plot(y_segmented, color='blue', label='Segmented Audio')
plt.title('Segmented Audio Example')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()
plt.show()


# # A4. Play around with your recorded speech signal for various segments. Understand the nature of the signal. Also observe with abruptly segmented speech, how the perception of that speech is affected.

# In[49]:


import sounddevice as sd
import time
filename = 'AISP_VOICE_21007.wav'
y, sr = librosa.load(filename)
segment_start = 0.2
segment_end = 1.0
segment = y[int(segment_start * sr):int(segment_end * sr)]
#sd.play(segment, sr)
#time.sleep(2)
ipd.Audio(segment, rate=sr)


# In[50]:


segment_start = 1.2
segment_end = 2.0
segment = y[int(segment_start * sr):int(segment_end * sr)]
#sd.play(segment, sr)
#time.sleep(2)
ipd.Audio(segment, rate=sr)


# In[48]:


print('Sample_rate:',sr)


# In[ ]:




