import tensorflow as tf
from tensorflow import keras
import librosa
from sklearn.model_selection import train_test_split

# NOTE: Since we are not displaying any graphs here we do not need librosa.display

# Sources: https://www.pythonpool.com/spectrogram-python/
#          https://www.codespeedy.com/determine-input-shape-in-keras-tensorflow/

# Path to full song audio file
full_audio = "SoundFiles\EpicSuspensefulDemo.wav"
# Using standard sample rate for an audio signal we load the audio
full_signal, full_sampleRate = librosa.load(full_audio, sr=44100)
# Apply short time fourier transforms on signal
full_shortTimeFourierTransforms = librosa.stft(full_signal)
# Convert to dB
STFTdbFullSong = librosa.amplitude_to_db(abs(full_shortTimeFourierTransforms))
print('First song done processing')

# Path to a single track of the audio file
# Path to full song audio file
track_audio = "SoundFiles\EpicSuspensefulTracks_Drum_Hit.wav"
# Using standard sample rate for an audio signal we load the audio
track_signal, track_sampleRate = librosa.load(track_audio, sr=44100)
# Apply short time fourier transforms on signal
track_shortTimeFourierTransforms = librosa.stft(track_signal)
# Convert to dB
STFTdbTrack = librosa.amplitude_to_db(abs(track_shortTimeFourierTransforms))
print('Second song done processing')

# Split data into test and training sets
x_train, x_test, y_train, y_test = train_test_split(STFTdbFullSong, STFTdbTrack, test_size=0.5, shuffle=False)

# Shallow network for testing tensorflow
shape = x_train.shape
inputs = tf.keras.Input(shape=shape[1])
hidden_layers = tf.keras.layers.Dense(12, activation=tf.nn.relu)(inputs)
output = tf.keras.layers.Dense(1, activation=tf.nn.softmax)(hidden_layers)
model = tf.keras.Model(inputs=inputs, outputs=output)
model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
hist = model.fit(x_train, y_train,
    batch_size = 50,
    epochs = 50,
    verbose = 1,
    validation_split = 0.4)

model_test = model.evaluate(x_test, y_test, verbose=2)

# Changing the accuracy into a percentage
testing_acc = model_test[1]
# Printing the accuracy
print('Test Accuracy: ', testing_acc,'%')