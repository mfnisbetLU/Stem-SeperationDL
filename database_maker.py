# Import declerations
import os
import pandas as pd

# Root path is the source folder, other declerations are temp lists and the dataframe decleration
root_path = 'Stem-SeperationDL\SoundFilesPrep'
temp_list = []
temp_list1 = []
dataframe_list = []
df = pd.DataFrame(columns=['Track','bass','drums','mixture','other','vocals'])
df.set_index(['Track'])

# Walk through directories and add to dataframe
for(roots, dirs, files) in os.walk(root_path):
    # The first value of the list will be the registry name (currently using as index)
    temp_list.append(roots)
    temp_list1 = files
    # Append each file in the registry to the reigstry name to convert to tuple
    for i in temp_list1:
        temp_list.append(i)
    
    # Append tuples to list to make a list of tuples
    temp_tuple = tuple(temp_list)
    dataframe_list.append(temp_tuple)
    
    # Clear out the temp lists
    temp_list = []
    temp_list1 = []

# Make a dataframe based on the list of tuples
df = pd.DataFrame(dataframe_list, columns = ['Track', 'Bass', 'Drums', 'Mixture', 'Other', 'Vocals'])
# The 'track' file is the index
df.set_index('Track')
print(df)