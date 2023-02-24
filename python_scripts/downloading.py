# This file declares functions I will use to download and clean my data. This script is called by the main jupyter notebook in this repo.
import os
import requests
import pandas as pd
import kaggle
# We first need to pip install kaggle for the following commands to work
kaggle.api.authenticate()

## Function declaration

def download_friends_script():
    """
    This function downloads the Friends tv show script from the "Friends TV Show Script" from Kaggle using the Kaggle API.
    """
    os.system('kaggle datasets download -d divyansh22/friends-tv-show-script --unzip --p "data/"')
    
    return "Done"