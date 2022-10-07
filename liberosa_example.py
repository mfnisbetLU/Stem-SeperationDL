import os
from platform import python_version

import matplotlib.pyplot as plt
 
import librosa
import librosa.display

if python_version == '3.7.1':

    # Sources: https://www.pythonpool.com/spectrogram-python/
    # NOTE: Runs in Python 3.7.1

    # Path to audio file
    audio = "C:\\Users\\russa\\OneDrive\\Documents\\GitHub\\Stem-SeperationDL\\ssriduvvuri_C_Minor_Chord_Progression_1451.wav"

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
    plt.show()
