# This is where the main function and GUI should probably go
# Currently use as a place to test functions

# Import functions from another file
from generate_dataframe import *
from batch_preprocess import *

# Process all the audio files then generate a dataframe
batch_preprocess()
df = generate_dataframe()
print(df)