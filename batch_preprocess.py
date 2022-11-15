'''
This function makes a new folder for each existing folder in the SoundFiles directory, then calls the audio_preprocess function
and puts them into their respective folders, ready for model training
'''
import os
import shutil
from audio_preprocess import *

# Paths of the operation, root path is old folder parent is parent folder new path is new folder

root_path = 'Stem-SeperationDL\SoundFiles'
new_path = 'Stem-SeperationDL\SoundFilesPrep'

def batch_preprocess():
    # Prints each directors and each file in the directory
    for (root,dirs,files) in os.walk(root_path):
        print (root)
        print (files)
        print ('--------------------------------')

    # !!WARNING!!
    # Deletes the SoundFilesPrep folder and then creates a copy of the SoundFiles folder
    shutil.rmtree(new_path)
    destination = shutil.copytree(root_path,new_path)

    # Walks through the directories
    for(subdir, dirs, files) in os.walk(new_path):
        # For each file in the directory
        for file in files:
            # File path
            file_path = subdir + "\\" + file
            print(file_path)
            # New file is the file with preprocessing from the file path
            new_file = audio_preprocess(file_path,0,10000)
            # Export and replace the old file with the new one
            new_file.export(file_path, format="wav")

        
        