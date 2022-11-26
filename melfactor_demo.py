'''
Generates the wave file and frequency graphs
Currently unused but could be utilized for visual demonstration of outputs
'''
# Geneartes a waveform and fequency chart for a wave file
import wave
import librosa
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

# Definitions
Test_File="Stem-SeperationDL\TestFilesPrep\Al James - Schoolboy Facination\\vocals.wav"
Test_Wav = wave.open(Test_File, 'rb')
Test_Freq = Test_Wav.getframerate()
n_samples = Test_Wav.getnframes()
t_audio = n_samples/Test_Freq
n_channels = Test_Wav.getnchannels()
signal_wave = Test_Wav.readframes(n_samples)

# Splits into left and right channels for ease of access
signal_array = np.frombuffer(signal_wave, dtype=np.int16)
l_channel = signal_array[0::2]
r_channel = signal_array[1::2]
times = np.linspace(0, n_samples/Test_Freq, num=n_samples)

# Plots the wave file
plt.figure(figsize=(15, 5))
plt.plot(times, l_channel)
plt.title('Left Channel')
plt.ylabel('Signal Value')
plt.xlabel('Time (s)')
plt.xlim(0, t_audio)
plt.show()

# Plots the spectrogram
plt.figure(figsize=(15, 5))
plt.specgram(l_channel, Fs=Test_Freq, vmin=-20, vmax=50)
plt.title('Left Channel')
plt.ylabel('Frequency (Hz)') 
plt.xlabel('Time (s)')
plt.xlim(0, t_audio)
plt.colorbar()
plt.show()
