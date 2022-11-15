import tensorflow as tf
from tensorflow import keras
from keras import callbacks
from database_maker import generate_dataframe
import librosa # for audio processing
import numpy as np

# NOTE: Since we are not displaying any graphs here we do not need librosa.display

# Sources: https://www.pythonpool.com/spectrogram-python/
#          https://www.codespeedy.com/determine-input-shape-in-keras-tensorflow/


# Generate dataframe of audio file locations with database_maker function
df = generate_dataframe()


# Retrieve the mix and bass for training
mix_df = df[['Track', 'Mixture']]
bass_df = df[['Track', 'Bass']]
print('Dataframes Generated')

# Create arrays of locations of files from the database
mix_location_arr = [[]]
for i, j in mix_df.iterrows():
    if i > 0:
        mix_location_arr.append(str(df.loc[i][0] + '/' + df.loc[i][1]))

bass_location_arr = [[]]
for i, j in bass_df.iterrows():
    if i > 0:
        bass_location_arr.append(str(df.loc[i][0] + '/' + df.loc[i][1]))


# Load audio files into an array
mix_audio_arr = []
for i in range(len(mix_location_arr)):
    if i > 0:
        temp, _ = librosa.load(mix_location_arr[i])
        mix_audio_arr.append(temp)

bass_audio_arr = []
for i in range(len(bass_location_arr)):
    if i > 0:
        temp, _ = librosa.load(bass_location_arr[i])
        bass_audio_arr.append(temp)
print('Files loaded')


# Find STFT of given audio
mix_stft_arr = []
for i in range(len(mix_audio_arr)):
    mix_stft_arr.append(librosa.stft(mix_audio_arr[i]))

bass_stft_arr = []
for i in range(len(bass_audio_arr)):
    bass_stft_arr.append(librosa.stft(bass_audio_arr[i]))
print('STFT done')


# Convert frequency in STFT to dB
mix_arr = [[]]
for i in range(len(mix_stft_arr)):
    mix_arr.append(librosa.amplitude_to_db(abs(mix_stft_arr[i])))

bass_arr = [[]]
for i in range(len(bass_stft_arr)):
    bass_arr.append(librosa.amplitude_to_db(abs(bass_stft_arr[i])))
print('Conversion to dB done')


# Split data into test and train sets
mix_arr_test = []
mix_arr_train = []
for i in range(len(mix_arr)):
    temp_test, temp_train = np.array_split(mix_arr[i], 2)
    mix_arr_test.append(temp_test) 
    mix_arr_train.append(temp_train)

bass_arr_test = [[]]
bass_arr_train = [[]]
for i in range(len(bass_arr)):
    temp_test, temp_train = np.array_split(bass_arr[i], 2)
    bass_arr_test.append(temp_test) 
    bass_arr_train.append(temp_train)
print('Data split into test and train sets')

'''
# Normalize data
mix_arr_test = tf.keras.utils.to_categorical(mix_arr_test/(np.linalg.norm(mix_arr_test)))
mix_arr_train = tf.keras.utils.to_categorical(mix_arr_train/(np.linalg.norm(mix_arr_train)))
bass_arr_test = bass_arr_test/(np.linalg.norm(bass_arr_test))
bass_arr_train = bass_arr_train/(np.linalg.norm(bass_arr_train))
print('Data normalized')
print('Test:', mix_arr_test.shape)
print('Train:', mix_arr_train.shape)
'''
for i in range(len(mix_arr)):
    mix_arr_test[i] = mix_arr_test[i]/(np.linalg.norm(mix_arr_test[i]))
    mix_arr_train[i] = mix_arr_train[i]/(np.linalg.norm(mix_arr_train[i]))

for i in range(len(bass_arr)):
    bass_arr_test[i] = bass_arr_test[i]/(np.linalg.norm(bass_arr_test[i]))
    bass_arr_train[i] = bass_arr_train[i]/(np.linalg.norm(bass_arr_train[i]))
print('Data normalized')


# Set up data as np arrays
mix_arr_test = np.asanyarray(mix_arr_test, dtype=object)
mix_arr_train = np.asanyarray(mix_arr_train, dtype=object)
bass_arr_test = np.asanyarray(bass_arr_test, dtype=object)
bass_arr_train = np.asanyarray(bass_arr_train, dtype=object)
print('Data converted to Numpy arrays\n\n\n')
print('Test Shape:', mix_arr_test[1].shape)
print('Train Shape:', mix_arr_train[1].shape)
print(mix_arr_test)
# Set early stopping
earlystopping = callbacks.EarlyStopping(monitor ="accuracy", 
                                        mode ="min", patience = 5, 
                                        restore_best_weights = True)




print("Creating Model")                                        
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(4, (3,3), input_shape=[512,1292,1]),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(4,(3,3), input_shape=[255,645,4]),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(4,(3,3), input_shape=[126,321,4]),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])  
model.summary()

# Compile model with stocastic gradient descent
model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["mae", "accuracy"])

print(mix_arr_train[1])
tf.config.run_functions_eagerly(True)
hist = model.fit(list(mix_arr_train), list(bass_arr_train),
    batch_size = 1,
    epochs = 100,
    verbose = 1,
    validation_data = (list(mix_arr_test), list(bass_arr_test)),
    callbacks=[earlystopping]
)

# Printing the accuracy
model_test = model.evaluate(mix_arr_test[1], mix_arr_test[1], verbose=2)

print(f" Model mse, mae and accuracy: {model_test}")

TrackPredictionMask=model.predict(mix_arr_test[1])
print("Pred", TrackPredictionMask)
print("Og", mix_arr_test[1])
print("Masked Test", (mix_arr_test[1]*TrackPredictionMask))