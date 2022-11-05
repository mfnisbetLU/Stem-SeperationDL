!apt install ffmpeg
pip install stempeg
import stempeg
import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import librosa
import librosa.display

L, rate = librosa.load("A Classic Education - NightOwl.stem.mp4")

audio, sample_rate = stempeg.read_stems("A Classic Education - NightOwl.stem.mp4",stem_id=None)
audio.shape

@property
def title_streams(self):
    """Returns stream titles for all substreams"""
    return [
        stream['tags'].get('handler_name')
        for stream in self.audio_streams
    ]
  
plt.boxplot(audio[0])
# show plot
plt.show()

L
audio[0, :, 0] # 3d arrays 
r = np.ptp(audio[0],axis=1)
r

plt.figure(figsize=(5, 5))
plt.title('Waveform comparison of librosa and stempeg')
librosa.display.waveshow(L, sr=rate)
librosa.display.waveshow(audio[0, :, 1], sr=sample_rate)


shortTimeFourierTransforms = librosa.stft(L)
# Convert to dB
STFTdb = librosa.amplitude_to_db(abs(shortTimeFourierTransforms))

plt.figure(figsize=(5, 5))
librosa.display.specshow(STFTdb, sr=rate, x_axis='time', y_axis='hz')
plt.colorbar()
plt.title('Volume of Frequency over Time (dB)')
plt.show()
