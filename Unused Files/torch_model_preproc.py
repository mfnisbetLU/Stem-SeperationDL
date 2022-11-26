'''
This is a test model using the pytorch library to handle transformations
Code based off the tutorial from here: https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5
Currently unused
'''
import tensorflow as tf
from tensorflow import keras
from keras import callbacks
from generate_dataframe import generate_dataframe
import librosa # for audio processing
import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
from batch_preprocess import *
import math, random
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset, random_split
from torchaudio import transforms
from IPython.display import audio
from torch.utils.data import random_split


# Process the data then generate a model from it
# Can be ignored if tweaking the model
# batch_preprocess()
df = generate_dataframe()
# Don't need the first row of the dataframe on new generation
df = df.drop(df.index[0])

# Loads an audio file
class AudioUtil():
    @staticmethod 
    def open(song):
        sig, sr = torchaudio.load(song)
        return (sig, sr)
    
# Shift the signal left or right by some percent for training
@staticmethod
def time_shift(song, shift_limit):
    sig,sr = song
    _, sig_len = sig.shape
    shift_amt = int(random.random() * shift_limit * sig_len)
    return (sig.roll(shift_amt),sr)

# Converts a song to a spectrogram
def spectro_gram(song, n_mels=64, n_fft=1024, hop_len=None):
    sig,sr = song
    top_db = 80
    spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return(spec)

# Function returns an augmented spectogram for additional testing
@staticmethod
def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
    _, n_mels, n_steps = spec.shape
    mask_value = spec.mean()
    aug_spec = spec
    freq_mask_param = max_mask_pct * n_mels
    
    # Apply frequency masking
    for _ in range(n_freq_masks):
        aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)
    
    # Apply time masking
    time_mask_param = max_mask_pct * n_steps
    for _ in range(n_time_masks):
        aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)
        
    return aug_spec

'''
Creating a class for the sound dataset, works similar to a java class 
'''
class SoundDS(Dataset):
    
    # Sound dataset definitions
    def init(self, df, data_path):
        self.df = df
        self.data_path = str(data_path)
        self.duration = 10000
        self.sr = 44100
        self.channel = 2
        self.shift_pact = .4
        
    # Number of items in dataset
    def len(self):
        return len(self.df)
    
    # Gets the track using the path and track name
    def getitem(self, id):
        audio_file = self.data_path + self.df.loc[id, 'relative_path']
        class_id = self.df.loc[id, 'Track']

        # Calls the functions to transform the track
        song = AudioUtil.open(audio_file)
        shift_song = AudioUtil.time_shift(song, self.shift_pct)
        spec_song = AudioUtil.spectro_gram(shift_song, n_mels=64, n_fft=1024, hop_len=None)
        augspec_song = AudioUtil.spectro_augment(spec_song, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
        
        return augspec_song, class_id

myds = SoundDS(df)

# Split into test and training sets
num_items = len(myds)
num_train = round(num_items*.8)
num_val = num_items-num_train
train_ds, val_ds = random_split(myds, [num_train, num_val])

# Final train and test (validation) test sets
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)