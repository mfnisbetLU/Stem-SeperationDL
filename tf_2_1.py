import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from generate_dataframe import generate_dataframe
import librosa 
import numpy as np
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime 
from audio_preprocess import *
'''
NOTE: Since we are not displaying any graphs here we do not need librosa.display
features_extractor and model based off code from here: https://www.analyticsvidhya.com/blog/2022/03/implementing-audio-classification-project-using-deep-learning/
Useful function: print(df.to_string()) prints every value without ellipses shortening       

'''
# Generate dataframe of audio file locations with generate_dataframe function
# The first value is just the parent folder for some reason so drop that
old_df = generate_dataframe()
old_df = old_df.drop(old_df.index[0])

# Create an array that holds the location to a track based on the generated dataframe
bass_location_arr = []
for i, j in old_df.iterrows():
    if i > 0:
        bass_location_arr.append(str(old_df.loc[i][0] + '/' + old_df.loc[i][1]))

# Set that location to be the 'location' and the 'class' to be the instrument
# This is repeated for each instrument
bass_df = pd.DataFrame(columns=['Location','Class'])
bass_df['Location'] = bass_location_arr
bass_df['Class'] = "Bass"

drum_location_arr = []
for i, j in old_df.iterrows():
    if i > 0:
        drum_location_arr.append(str(old_df.loc[i][0] + '/' + old_df.loc[i][2]))

drum_df = pd.DataFrame(columns=['Location','Class'])
drum_df['Location'] = drum_location_arr
drum_df['Class'] = "Drum"


mix_location_arr = []
for i, j in old_df.iterrows():
    if i > 0:
        mix_location_arr.append(str(old_df.loc[i][0] + '/' + old_df.loc[i][3]))

mix_df = pd.DataFrame(columns=['Location','Class'])
mix_df['Location'] = mix_location_arr
mix_df['Class'] = "Mix"

other_location_arr = []
for i, j in old_df.iterrows():
    if i > 0:
        other_location_arr.append(str(old_df.loc[i][0] + '/' + old_df.loc[i][4]))

other_df = pd.DataFrame(columns=['Location','Class'])
other_df['Location'] = other_location_arr
other_df['Class'] = "Other"


voc_location_arr = []
for i, j in old_df.iterrows():
    if i > 0:
        voc_location_arr.append(str(old_df.loc[i][0] + '/' + old_df.loc[i][5]))

voc_df = pd.DataFrame(columns=['Location','Class'])
voc_df['Location'] = voc_location_arr
voc_df['Class'] = "Vocals"

# Concatenate all the dataframes into one large vertical dataframe
new_df = pd.concat([bass_df,drum_df,mix_df,other_df,voc_df], ignore_index=True)

# Gets the mel frequency cepstral coefficient for each file
def features_extractor(file):
    # Loads the audio file
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast') 
    # Extracts the mfcc
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    # Mean transpose of the scaled features
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features

# Makes a new dataframe out of the MFCC features and the original class
mfcc_df = pd.DataFrame(columns=['Feature','Class'])
mfcc_df['Feature'] = new_df['Location'].apply(features_extractor)
mfcc_df['Class'] = new_df['Class']

# Split the dataset into features and classes
X=np.array(mfcc_df['Feature'].tolist())
y=np.array(mfcc_df['Class'].tolist())
# Label Encoding 
labelencoder=LabelEncoder()
# Fit the label encoder
y= keras.utils.to_categorical(labelencoder.fit_transform(y))
# Split into test and training sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# Number of classes (automatically calculates based on number of categories)
num_labels=y.shape[1]
input_shape=X_train.shape

print(y_train.shape)
print(input_shape)
model=Sequential()
model.add(tf.keras.layers.Input(shape=(40,)))
model.add(tf.keras.layers.RepeatVector(5))
model.add(tf.keras.layers.Conv1D(128, (3)))
model.add(tf.keras.layers.MaxPool1D(pool_size=(2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(40, activation=tf.nn.elu))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(5, activation=tf.nn.softmax))
model.summary()

# Compile and summarize the model
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

# Training the model, 200 epochs seems to work better than 100
num_epochs = 200
num_batch_size = 25
# Using the model checkpoint to save the best values
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='./audio_classification.hdf5', 
                               verbose=1, save_best_only=True)
# Using datetime to calculate how long the training took place
start = datetime.now()

# Fits the model for the first time
model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)
duration = datetime.now() - start
print("Training completed in time: ", duration)

# Tests the accuracy of the model
test_accuracy=model.evaluate(X_test,y_test,verbose=0)

#model.predict_classes(X_test)
predict_x=model.predict(X_test) 
classes_x=np.argmax(predict_x,axis=1)
# Prints the classes predicted by the model
print(classes_x)

# Print test accuracy
print(test_accuracy[1])




# Saves the model (currently can't load due to keras issue)
tf.saved_model.save(model, 'Stem-SeperationDL/model/')

'''
---------------------------------------------------
TESTING 
There's some known issues with saving/loading models
Even though there are save/load functions keras has not resolved the issue yet
So the testing must be done in the same file
'''
# Import the test file
Test_File="TestFilesPrep/Arise - Run Run Run/drums.wav"
# Process audio and convert to mfcc like above
audio, sample_rate = librosa.load(Test_File, res_type='kaiser_fast') 
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

# Reshape MFCC feature to 2-D array
mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)

# Predict the label using th emodel
x_predict=model.predict(mfccs_scaled_features) 
predicted_label=np.argmax(x_predict,axis=1)
print(predicted_label)
prediction_class = labelencoder.inverse_transform(predicted_label) 
print(prediction_class)
