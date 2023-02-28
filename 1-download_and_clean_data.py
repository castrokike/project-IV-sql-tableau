# This script will be used to download and process the data from the Friends TV show script.
# It calls functions declared in the downloading_and_cleaning_func.py so the processed showed here is very summaraized.

import os
import requests
import pandas as pd
import kaggle
import numpy as np
from python_scripts import downloading_and_cleaning_func as d_c

# First we authenticate using the kaggle API.
kaggle.api.authenticate()

# We use a function I delcared to download the Friends script as a .txt file.
d_c.download_friends_script()

# Now we read the text file using open() so that we can turn it into a list of strings to process with the a function we defined.
with open("data/Friends_Transcript.txt", "r") as f:
    script = f.readlines()

# Using a cleaning function we process the file so that we turn the .txt into a Pandas DF:
friends_script = d_c.process_script(script)

# Using several functions we clean the script in order to produce 3 DataFrames with information about the script, scenes/episodes and episodes/season.
friends_script, f_scene_info, f_seasons = d_c.clean_script(friends_script)

# Exports these Dataframes to CSV files so that they can be uploades to SQL later using the uploading_to_sql.py script.
d_c.export_friends_info_csv(friends_script, f_seasons, f_scene_info)