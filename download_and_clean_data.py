import os
import requests
import pandas as pd
import kaggle
import numpy as np
from python_scripts import downloading_and_cleaning_func as d_c

# First we authenticate using the kaggle API.
#kaggle.api.authenticate()

# We use a function I delcared to download the Friends script as a .txt file.
#d_c.download_friends_script()

# Now we read the text file using open().
with open("data/Friends_Transcript.txt", "r") as f:
    script = f.readlines()

# Using a cleaning function we process the file so that we turn the .txt into a Pandas DF:
friends_script = d_c.process_script(script)

friends_script, f_scene_info, f_seasons = d_c.clean_script(friends_script)

d_c.export_friends_info_csv(friends_script, f_seasons, f_scene_info)