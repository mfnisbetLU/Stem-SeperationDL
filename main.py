# This is where the main function and GUI should probably go
# Currently use as a place to test functions

# Import functions from another file
from generate_dataframe import *
from batch_preprocess import *
import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

# Process all the audio files then generate a dataframe
# Don't need to do this every time
# !!!WARNING!!! Deletes files in new folder, then copies from old folder, if old folder is empty it will only delete
batch_preprocess()
batch_preprocess_test()
# Generates the dataframe
df = generate_dataframe()
# Don't need the first value since it's the directory and we've included that
df = df.drop(df.index[0])
print(df)

