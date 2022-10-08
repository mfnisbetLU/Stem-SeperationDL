import os
from platform import python_version

import matplotlib.pyplot as plt
 
import librosa
import librosa.display

# Sources: https://www.pythonpool.com/spectrogram-python/

# Path to audio file
audio = "E:\Code\DSIP_Project\Stem-SeperationDL\EpicSuspensefulDemo.wav"

# Using standard sample rate for an audio signal we load the audio
signal, sampleRate = librosa.load(audio, sr=44100)

# Display signal as a waveform (amplitude over time) using librosa
plt.figure(figsize=(5, 5))
librosa.display.waveshow(signal, sr=sampleRate)

# Apply short time fourier transforms on signal
shortTimeFourierTransforms = librosa.stft(signal)
# Convert to dB
STFTdb = librosa.amplitude_to_db(abs(shortTimeFourierTransforms))
# Use librosa to display spectrogram
plt.figure(figsize=(5, 5))
librosa.display.specshow(STFTdb, sr=sampleRate, x_axis='time', y_axis='hz')
plt.colorbar()
plt.title('Volume of Frequency over Time (dB)')
plt.show()
