'''
This script takes a wave file, detects and removes any leading silence, normalizes it then slices it into a 30 second segment and returns that segment.
Code based on the answers found here:
https://stackoverflow.com/questions/29547218/remove-silence-at-the-beginning-and-at-the-end-of-wave-files-with-pydub
https://stackoverflow.com/questions/37999150/how-to-split-a-wav-file-into-multiple-wav-files
''' 
from pydub import AudioSegment, effects
from scipy.io import wavfile


def audio_preprocess(audio_file,trim_start,trim_end):

    # Path to the wav file track, can be adjusted
    sound = AudioSegment.from_file(audio_file, format="wav")
    # Defines the start and end of the silence
    start_trim = leading_silence(sound)
    # Defines the length of the silence
    duration = len(sound)    
    # Resulting sound file 
    trimmed_sound = sound[start_trim:duration]
    # Normalize the trimmed audio file
    normalized_sound = effects.normalize(trimmed_sound)
    # The new audio file trimmed from defined start and end
    new_audio = normalized_sound[trim_start:trim_end]
    return new_audio

# Function to detect the leading silence of a track
def leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    # Temp variable stores length of the trim
    trim_ms = 0 
    # Avoid infinite loop
    assert chunk_size > 0 
    # While the track is below the since threshold, start trimming the sound length 
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms