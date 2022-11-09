import tensorflow as tf
from tensorflow import keras
from keras import callbacks
import librosa # for audio processing
import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
# NOTE: Since we are not displaying any graphs here we do not need librosa.display

# Sources: https://www.pythonpool.com/spectrogram-python/
#          https://www.codespeedy.com/determine-input-shape-in-keras-tensorflow/




# Using standard sample rate for an audio signal we load the audio
full_audio, sr = librosa.load("SoundFiles\EpicSuspensefulDemo\EpicSuspensefulDemo.wav", sr = 44100)

# Apply short time fourier transforms on signal
full_shortTimeFourierTransforms = librosa.stft(full_audio)
# Convert to dB
STFTdbFullSong = librosa.amplitude_to_db(abs(full_shortTimeFourierTransforms))
print('First song done processing: STFTdbFullSong')
print(STFTdbFullSong)


# Using standard sample rate for an audio signal we load the audio
track_audio, sr = librosa.load("SoundFiles\EpicSuspensefulDemo\EpicSuspensefulTracks_Drum_Hit.wav", sr=44100)

# Apply short time fourier transforms on signal
track_shortTimeFourierTransforms = librosa.stft(track_audio)
# Convert to dB
STFTdbTrack = librosa.amplitude_to_db(abs(track_shortTimeFourierTransforms))
print('Second song done processing')
print(STFTdbTrack)

# Split data into test and train 
SongTest, SongTrain = np.array_split(STFTdbFullSong, 2)
SongTest = SongTest[:-1]

TrackTest, TrackTrain = np.array_split(STFTdbTrack, 2)
TrackTest = TrackTest[:-1]

# Reshape data for convolutional network
#SongTest = SongTest.reshape(-1,8,9598,1)
SongTrain = SongTrain.reshape(-1,8,9598,1)
#TrackTest = TrackTest.reshape(-1,8,9598,1)
TrackTrain = TrackTrain.reshape(-1, 8,9598,1)


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
    tf.keras.layers.Conv2D(32, (3,3), padding="same", input_shape=(8,9598,1),activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])  
model.summary()

# Compile model with stocastic gradient descent
model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["mae", "accuracy"])

hist = model.fit(SongTrain, TrackTrain,
    batch_size = 512,
    epochs = 100,
    verbose = 1,
    validation_data = (SongTest, TrackTest),
    callbacks=[earlystopping]
)

# Printing the accuracy
model_test = model.evaluate(SongTest, TrackTest, verbose=2)

print(f" Model mse, mae and accuracy: {model_test}")

TrackPredictionMask=model.predict(SongTest)
print("Pred", TrackPredictionMask)
print("Og", TrackTest)
print("Dif", ((SongTest-TrackPredictionMask)-TrackTest))