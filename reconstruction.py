from scipy.signal import stft, istft
import numpy as np

def reconstruct(spectrogram, sample_rate, nperseg, iters=100):
    length = istft(spectrogram, sample_rate, nperseg=nperseg)[1].shape[0]
    x = np.random.normal(size=length)
    for i in range(iters):
        # Code based on the answer here: https://dsp.stackexchange.com/a/3410
        X = stft(x, sample_rate, nperseg=nperseg)[2]
        Z = spectrogram * np.exp(np.angle(X) * 1j)
        x = istft(Z, sample_rate, nperseg=nperseg)[1]
    return x 
