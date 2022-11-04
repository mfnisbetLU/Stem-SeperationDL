import tensorflow as tf
from tensorflow import keras
import librosa # for audio processing
from sklearn.model_selection import train_test_split
from keras import callbacks
import numpy as np

# NOTE: Since we are not displaying any graphs here we do not need librosa.display

# Sources: https://www.pythonpool.com/spectrogram-python/
#          https://www.codespeedy.com/determine-input-shape-in-keras-tensorflow/




# Using standard sample rate for an audio signal we load the audio
full_audio, sr = librosa.load("SoundFiles\EpicSuspensefulDemo.wav", sr = 44100)

# Apply short time fourier transforms on signal
full_shortTimeFourierTransforms = librosa.stft(full_audio)
# Convert to dB
STFTdbFullSong = librosa.amplitude_to_db(abs(full_shortTimeFourierTransforms))
print('First song done processing')
print(STFTdbFullSong)


# Using standard sample rate for an audio signal we load the audio
track_audio, sr = librosa.load("SoundFiles\EpicSuspensefulTracks_Drum_Hit.wav", sr=44100)

# Apply short time fourier transforms on signal
track_shortTimeFourierTransforms = librosa.stft(track_audio)
# Convert to dB
STFTdbTrack = librosa.amplitude_to_db(abs(track_shortTimeFourierTransforms))
print('Second song done processing')
print(STFTdbTrack)

# Split data into test, train 
SongTest, SongTrain = np.array_split(STFTdbFullSong, 2)
print("Song train", SongTrain.shape)
TrackTest, TrackTrain = np.array_split(STFTdbTrack, 2)
print("Track train", TrackTrain.shape)


# Normalize data
SongTest = SongTest/(np.linalg.norm(SongTest))
SongTrain = SongTrain/(np.linalg.norm(SongTrain))
TrackTest = TrackTest/(np.linalg.norm(TrackTest))
TrackTrain = TrackTrain/(np.linalg.norm(TrackTrain))


# Set early stopping
earlystopping = callbacks.EarlyStopping(monitor ="accuracy", 
                                        mode ="min", patience = 5, 
                                        restore_best_weights = True)
print("Creating Model")                                        
model = tf.keras.Sequential([
    # Creation of a 1 dimensional convolutional network
    tf.keras.layers.Dense(288, activation = tf.nn.relu, 
    input_shape=SongTrain.shape[1:]),
    tf.keras.layers.Dense(288, activation = tf.nn.relu),
    tf.keras.layers.Dense(TrackTrain.shape[1], activation=tf.nn.softmax)
])  
model.summary()

# Compile model with stocastic gradient descent
model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["mae", "accuracy"])


hist = model.fit(SongTrain, TrackTrain,
    batch_size = 512,
    epochs = 100,
    verbose = 1,
    validation_split = 0.4,
    callbacks=[earlystopping]
)

# Printing the accuracy
model_test = model.evaluate(SongTest, TrackTest, verbose=2)

print(f" Model mse, mae and accuracy: {model_test}")

TrackPrediction=model.predict(SongTest)
print("Pred", TrackPrediction)
print("Og", TrackTest)
print("Dif", (TrackPrediction-TrackTest))