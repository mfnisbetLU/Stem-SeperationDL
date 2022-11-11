# This is where the main function and GUI should probably go
# Currently use as a place to test functions

# Import functions from another file
from preprocess_slicer import *

audio_file = "Stem-SeperationDL\SoundFiles\Al James - Schoolboy Facination\mixture.wav"

new_file = audio_preprocess(audio_file,0,30000)

new_file.export("Stem-SeperationDL/NewMix.wav", format="wav")