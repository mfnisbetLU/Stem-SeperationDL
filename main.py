# This is where the main function and GUI should probably go
# Currently use as a place to test functions

# Import functions from another file
from generate_dataframe import *
from batch_preprocess import *

# Process all the audio files then generate a dataframe
# Don't need to do this every time
batch_preprocess()
# Generates the dataframe
df = generate_dataframe()
# Don't need the first value since it's the directory and we've included that
df = df.drop(df.index[0])
print(df)