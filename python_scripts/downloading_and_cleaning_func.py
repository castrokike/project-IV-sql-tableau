# This file declares functions I will use to download and clean my data. This script is called by the main jupyter notebook in this repo.
import os
import requests
import numpy as np
import pandas as pd
import kaggle
import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.downloader.download('vader_lexicon', quiet = True)
from textblob import TextBlob

# Muting warnings
pd.set_option('mode.chained_assignment', None)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

## Function declaration

def download_friends_script():
    """
    This function downloads the Friends tv show script from the "Friends TV Show Script" from Kaggle using the Kaggle API.
    """
    os.system('kaggle datasets download -d divyansh22/friends-tv-show-script --unzip --p "data/"')
    
    return "Done! Downloaded to data/"

def process_line(line):
    """The function process_line takes a line of text from the Friends script and extracts relevant information from it.
        It returns a tuple with four elements representing episode, scene, character, and dialogue, respectively.
        If the line contains scene information, it extracts the scene description and returns it along with empty strings for the other elements.
        If the line contains title information, it extracts the episode title and returns it along with empty strings for the other elements.
        If the line contains character information, it extracts the character name and dialogue and returns them along with empty strings for the other elements.
        If the line contains neither scene, title, nor character information, it returns an empty string for the episode and character elements and the line itself for the dialogue element.
        
        This function will be used to read the script and turn it into a DF.
        """
    # Remove newline character
    line = line.strip()
    # Check if line contains scene information
    if line.startswith("[Scene:"):
        scene = line[8:-2]
        return ("", scene, "none", "")
    # Check if line contains title information
    elif line.startswith("THE ONE"):
        episode = line.split("(")[0].strip()
        return (episode, "", "none", "")
    # Check if line contains character information
    elif line.strip() and line[0].isupper() and ":" in line:
        character, dialogue = line.split(":", 1)
        return ("", "", character, dialogue.strip())
    else:
        return ("", "", "none", line)
    
def process_script(script):
    """
    Process the given script of Friends TV show, extracting relevant information such as 
    the episode, scene, character, and dialogue for each line using the process_line function, and returns a pandas DataFrame 
    with columns for episode, scene, character, and line.
    
    Args:
    script: A list of strings where each line represents a line of dialogue from the script.
    
    Returns:
    A pandas DataFrame with columns for episode, scene, character, and line, where each row 
    represents a line of dialogue from the script with the corresponding extracted information.
    """
    lines = []
    for line in script:
        lines.append(process_line(line))
    return pd.DataFrame(lines, columns=["episode", "scene", "character", "line"])

def clean_friends_script(friends_script):
    """
    Cleans the Friends TV show script dataframe.

    Args:
    friends_script: pandas DataFrame containing the Friends TV show script data.

    Returns:
    A pandas DataFrame that has no missing values in columns "episode", "scene", "character" or "line".
    """
    
    # Replace empty string values with NaN so that they are easier to process in the future.
    friends_script.replace("", pd.NA, inplace=True)

    # Fill forward missing values in columns "episode" and "scene" so that we can process these in the future in SQL.
    friends_script["episode"].fillna(method="ffill", inplace=True)
    friends_script["scene"].fillna(method="ffill", inplace=True)

    # Drop rows where "character" column has value "none" and "line" column is NaN, these lines are just titles of the episodes and we already have this information in the rest of the cells of the "episode" column.
    friends_script = friends_script.loc[(friends_script['character'] != 'none') & (friends_script['line'].notna())]
    
    return friends_script

def create_scene_info(friends_script):
    """
    This function takes a cleaned Friends script and returns a new DataFrame with information on the scenes per episode.

    Parameters:
    friends_script (DataFrame): a cleaned Friends script DataFrame.

    Returns:
    f_scene_info (DataFrame): a DataFrame containing the episode title, scene number, and scene name for each scene in each episode.
    """

    # Select only the episode and scene columns from the script
    f_scene_info =friends_script[["episode","scene"]]

    # Remove any duplicate rows
    f_scene_info.drop_duplicates(inplace=True)

    # Remove any rows with missing values
    f_scene_info.dropna(inplace=True)

    # Reset the index of the DataFrame
    f_scene_info.reset_index(inplace=True)

    # Remove the old index column from the DataFrame
    f_scene_info.drop("index", axis =1, inplace=True)

    # Reset the index of the DataFrame to use scene numbers instead of row numbers
    f_scene_info.reset_index(inplace=True)

    # Rename the new index column to "scene_number"
    f_scene_info.rename(columns={"index":"scene_number"}, inplace=True)

    # Convert the episode titles to title case (capitalizing the first letter of each word)
    f_scene_info["episode"] = f_scene_info["episode"].str.title()

    # Return the new DataFrame with scene information
    return f_scene_info

def get_seasons():
    """
    This function retrieves information about the seasons of the Friends TV show from a Wikipedia page, 
    and preprocesses the data to be used for analysis.

    Returns:
    f_seasons (pandas DataFrame): A dataframe containing information about the seasons of the Friends TV show.
    """

    # URL of the Wikipedia page with the information about Friends seasons
    seasons_url="https://en.wikipedia.org/wiki/List_of_Friends_episodes"

    # Retrieve the HTML tables from the Wikipedia page
    seasons_list = pd.read_html(seasons_url)

    # Select only the tables with season information
    f_seasons = []
    for i in list(range(1,12)):
        f_seasons.append(seasons_list[i])

    # Add a column with the season number to each table and concatenate the tables
    for i, df in enumerate(f_seasons):
        df["season"] = f'{i+1}'

    f_seasons = pd.concat(f_seasons, ignore_index=True)

    # Replace missing values with -1 for easier analysis and convert columns with episode numbers to integers
    f_seasons["No.overall"] = f_seasons["No.overall"].fillna(-1)
    f_seasons["No.overall"] = f_seasons["No.overall"].astype(int)
    f_seasons["No. inseason"] = f_seasons["No. inseason"].fillna(-1)
    f_seasons["No. inseason"] = f_seasons["No. inseason"].astype(int)

    # Remove text in square brackets from each cell of the dataframe (these are actions taken by the characters or descriptions of the scene that do not contribute to our analysis)
    f_seasons = f_seasons.applymap(lambda x: x if pd.isnull(x) else re.sub(r'\[.*?\]', '', str(x)))

    # Remove double quotes from the Title column and capitalize each word to be matched with our existing episode names from the scripts
    f_seasons['Title'] = f_seasons['Title'].str.replace('"', '')
    f_seasons["Title"] = f_seasons["Title"].str.title()

    # Fix the title of the first episode of the series
    f_seasons.loc[0, 'Title'] = 'The One Where Monica Gets A New Roommate'

    # Return the processed dataframe
    return f_seasons

def match_episodes(f_scene_info, f_seasons):
    """
    Matches the episodes in the given 'f_scene_info' dataframe with their corresponding episodes in the 'f_seasons' 
    dataframe, based on the common column "episode". Returns the updated 'f_scene_info' dataframe.
    
    :param f_scene_info: Pandas dataframe containing information about the scenes in each episode, with the "episode" column corresponding to the episode number.
    
    :param f_seasons: Pandas dataframe containing information about the episodes in each season, with the "No.overall" column corresponding to the episode number.
    
    :return: Pandas dataframe with the same columns as the input 'f_scene_info' dataframe, but with the "episode" column updated to match the episode numbers in 'f_seasons'.
    """
    # Merge the data frames based on the common column "episode"
    merged_df = pd.merge(f_scene_info, f_seasons[['No.overall', 'Title', 'season']], left_on='episode', right_on='Title', how='left')
    
    # Replace the "episode" column with the "No.overall" column where it is available
    merged_df['episode'] = merged_df['No.overall'].fillna(merged_df['episode'])

    # Drop unnecessary columns
    merged_df.drop(['No.overall', 'Title'], axis=1, inplace=True)

    # Manually replace non matching episodes
    mapping = {'The One Where Monica Gets A New Roomate' : '1' ,
    'The One With Two Parts, Part 1' : '16' ,
    'The One With Two Parts, Part 2' : '17' ,
    "The One With Ross' New Girlfriend" : '25',
    'The One Where Mr. Heckles Dies' : '27' ,
    'The One With The Last' : '32' ,
    "The One Where No-One'S Ready" : '50' ,
    'The One Where Monica And Richard Are Friends' : '13' ,
    'The One The Morning After' : '64' ,
    "The One With A Chick. And A Duck" : '69' ,
    "The One Where Chandler Crosses A Line" : '80' ,
    "The One Where They'Re Gonna Party!" : '82' ,
    "The One With Phoebes Uterus" : '84' ,
    "The One With Rachels Crush" : '86' ,
    "The One With Joeys Dirty Day" : '87' ,
    "The One With Ross'S Wedding Parts I And Ii" : '96' ,
    "The One With Ross'S Wedding - Uncut Version" : '97' ,
    "The One Hundredth" : '100' ,
    "The One With All The Kips" : '102' ,
    "The One With The Thanksgiving Flashbacks" : '105' ,
    "The One Where Everyone Finds Out" : '111' ,
    "The One With A Cop" : '113' ,
    "The One With Rachel'S Inadvertant Kiss" : '114' ,
    "The One With The Ride Along" : '117' ,
    "The One With Rosss Denial" : '124' ,
    "The One With Joeys Porsche" : '126' ,
    "The One The Last Night" : '127' ,
    "The One With Rachels Sister" : '134' ,
    "The One Where Chandler Cant Cry" : '135' ,
    "The One With The Unagi" : '138' ,
    "The One With Joeys Fridge" : '140' ,
    "The One Where Ross Meets Elizabeths Dad" : '142' ,
    "The One Where Pauls The Man" : '143' ,
    "The One With Monicas Thunder" : '147' ,
    "The One With Rachels Book" : '148' ,
    "The One With Phoebes Cookies" : '149' ,
    "The One With Rachels Assistant" : '150' ,
    "The One With Rosss Book" : '153' ,
    "The One Where Chandler Doesnt Like Dogs" : '154' ,
    "The One With All The Cheesecake" : '157' ,
    "The One Where Theyre Up All Night" : '158' ,
    "The One Where Rosita Dies" : '159' ,
    "The One Where They All Turn Thirty" : '160' ,
    "The One With Joeys New Brain" : '161' ,
    "The One With The Truth About London" : '162' ,
    "The One With Joeys Award" : '164' ,
    "The One With Ross And Monicas Cousin" : '165' ,
    "The One With Rachels Big Kiss" : '166' ,
    "The One With Chandlers Dad" : '168' ,
    "The One With Chandler And Monicas Wedding" : '169' ,
    'The One After "I Do"' : '171' ,
    "The One With Monicas Boots" : '180' ,
    "The One With Ross' Big Step Forward" : '181' ,
    "The One In Massapequa" : '188' ,
    "The One With Joeys Interview" : '189' ,
}
    merged_df['episode'] = merged_df['episode'].replace(mapping)
    
    # Re-match the seasons
    
    # Drop the "season" column as it will be replaced
    merged_df.drop(['season'], axis=1, inplace=True)

    # Merge the data frames based on the common column "episode"
    merged_df = pd.merge(merged_df, f_seasons[['No.overall', 'season']], left_on='episode', right_on='No.overall', how='left')
    
    # Modify seasons df to separate episodes with two parts
    new_rows = [{'No.overall': '96', 'No. inseason': '23', 'Title': "The One With Ross'S Wedding", 'Directed by': 'Kevin S. Bright', 'Written by': 'Michael BorkowStory by\u200a: Jill Condon & Amy ToominTeleplay by\u200a: Shana Goldberg-Meehan & Scott Silveri', 'Original air date': 'May\xa07,\xa01998', 'Prod.code': '466623', 'U.S. viewers(millions)': '31.61','season': '4', 'Rating(18–49)': np.nan, 'Rating/share(18–49)': '16.7/49', 'Special No.': np.nan, 'U.S. viewersmillions': np.nan},
            {'No.overall': '97', 'No. inseason': '24', 'Title': "The One With Ross'S Wedding", 'Directed by': 'Kevin S. Bright', 'Written by': 'Michael BorkowStory by\u200a: Jill Condon & Amy ToominTeleplay by\u200a: Shana Goldberg-Meehan & Scott Silveri', 'Original air date': 'May\xa07,\xa01998', 'Prod.code': '466624', 'U.S. viewers(millions)': '31.61','season': '4', 'Rating(18–49)': np.nan, 'Rating/share(18–49)': '16.7/49', 'Special No.': np.nan, 'U.S. viewersmillions': np.nan},
            {'No.overall': '16', 'No. inseason': '16', 'Title': 'The One With Two Parts', 'Directed by': 'Michael Lembeck', 'Written by': 'Marta Kauffman & David Crane', 'Original air date': 'February\xa023,\xa01995', 'Prod.code': '456665', 'U.S. viewers(millions)': '26.130.5', 'season': '1', 'Rating(18–49)': np.nan, 'Rating/share(18–49)': np.nan, 'Special No.': np.nan, 'U.S. viewersmillions': np.nan},
            {'No.overall': '17', 'No. inseason': '17', 'Title': 'The One With Two Parts', 'Directed by': 'Michael Lembeck', 'Written by': 'Marta Kauffman & David Crane', 'Original air date': 'February\xa023,\xa01995', 'Prod.code': '456666', 'U.S. viewers(millions)': '26.130.5', 'season': '1', 'Rating(18–49)': np.nan, 'Rating/share(18–49)': np.nan, 'Special No.': np.nan, 'U.S. viewersmillions': np.nan},
            {'No.overall': '169', 'No. inseason': '23', 'Title': "The One With Monica And Chandler'S Wedding", 'Directed by': 'Kevin S. Bright', 'Written by': 'Gregory S. MalinsMarta Kauffman & David Crane', 'Original air date': 'May\xa017,\xa02001', 'Prod.code': '226422', 'U.S. viewers(millions)': '30.05', 'season': '7', 'Rating(18–49)': np.nan, 'Rating/share(18–49)': '15.7/43', 'Special No.': np.nan, 'U.S. viewersmillions': np.nan},
            {'No.overall': '170', 'No. inseason': '24', 'Title': "The One With Monica And Chandler'S Wedding", 'Directed by': 'Kevin S. Bright', 'Written by': 'Gregory S. MalinsMarta Kauffman & David Crane', 'Original air date': 'May\xa017,\xa02001', 'Prod.code': '226423', 'U.S. viewers(millions)': '30.05', 'season': '7', 'Rating(18–49)': np.nan, 'Rating/share(18–49)': '15.7/43', 'Special No.': np.nan, 'U.S. viewersmillions': np.nan}]
    f_seasons = f_seasons.append(new_rows, ignore_index=True)
    f_seasons = f_seasons.drop(index=[93,15,162])

    # Drop the "season" column as it will be replaced
    merged_df.drop(['season'], axis=1, inplace=True)

    # Merge the data frames based on the common column "episode"
    merged_df = pd.merge(merged_df, f_seasons[['No.overall', 'season']], left_on='episode', right_on='No.overall', how='left')
    
    # Update f_scene_info with the matched episodes and drop unnecessary columns
    f_scene_info = merged_df.drop(["season", "No.overall_x", "No.overall_y"], axis=1)

    # Return updated dataframes
    return f_scene_info, f_seasons

def match_episode_numbers_in_script(friends_script,f_scene_info):
    """
    Matches the episodes in the given 'friends_script' dataframe with their corresponding episodes in the 'f_scene_info' 
    dataframe, based on the common column "scene". Returns the updated 'friends_script' dataframe.
    
    :param friends_script: Pandas dataframe containing the script for the Friends TV show, with the "scene" column corresponding to the scene number.
    
    :param f_scene_info: Pandas dataframe containing information about the scenes in each episode, with the "episode" column corresponding to the episode number.
    
    :return: Pandas dataframe with the same columns as the input 'friends_script' dataframe, but with the "episode" column updated to match the episode numbers in 'f_scene_info'.
    """
    # Drop rows with missing scene data, then titlecase episode column
    friends_script = friends_script.dropna(subset=['scene'])
    friends_script['episode'] = friends_script['episode'].str.title()

    # Merge the data frames based on the common column "scene"
    merged_df_scenes = pd.merge(friends_script, f_scene_info[['scene_number', 'episode','scene']], left_on='scene', right_on='scene', how='left')
    
    # Drop unnecessary columns
    merged_df_scenes.drop(["episode_x", "scene", "episode_y"], axis=1, inplace=True)
    
    # Update friends_script with merged data
    friends_script = merged_df_scenes

    return friends_script

def process_character_names(friends_script):
    """
    Processes character names in the given 'friends_script' dataframe, standardizing their format and removing any 
    unnecessary information. Returns the updated 'friends_script' dataframe.
    
    :param friends_script: Pandas dataframe containing information about each line in the Friends TV show script, including the character speaking.
    
    :return: Pandas dataframe with the same columns as the input 'friends_script' dataframe, but with standardized character names.
    """
    # Replace character names that include actions or descriptions
    friends_script["character"] = friends_script["character"].str.replace(r"^(Ross|Rachel|Monica|Joey|Chandler|Phoebe) \(.+", r"\1", regex=True)

    # Replace character names that have a single space after the name
    friends_script["character"] = friends_script["character"].str.replace(r"^(Ross|Rachel|Monica|Joey|Chandler|Phoebe) $", r"\1", regex=True)
    
    # Replace all-uppercase character names with standardized capitalization
    uppercase_names = {"MONICA": "Monica", "CHANDLER": "Chandler", "JOEY": "Joey", "PHOEBE": "Phoebe", "RACHELL": "Rachel", "ROSS": "Ross"}
    friends_script["character"] = friends_script["character"].replace(uppercase_names, regex=True)

    # Replace abbreviated character names with their full names
    abrev_names = {"MNCA": "Monica", "CHAN": "Chandler", "JOEY": "Joey", "PHOE": "Phoebe", "RACH": "Rachel", "ROSS": "Ross"}
    friends_script["character"] = friends_script["character"].replace(abrev_names, regex=True)

    return friends_script

# Initialize the SentimentIntensityAnalyzer outside the function as this function is used in an apply and will slow down the process considerably.
sia = SentimentIntensityAnalyzer()

def analyze_sentiment_sia(row):
    """
    Applies sentiment analysis to the 'line' column of a pandas DataFrame row using the SentimentIntensityAnalyzer from the nltk library, and returns a new pandas Series containing the polarity scores for the input row.
    This function will be used with the sentiment_analysis function to apply sentiment analysis to every line of the Friends TV show script.

    Parameters:
    ----------
    row : pandas DataFrame row
        The row to apply sentiment analysis to. The row must contain a 'line' column that contains text to be analyzed.
        
    Returns:
    -------
    pandas Series
        A pandas Series containing the following columns:
        - neg : float
            The negative sentiment score for the text in the 'line' column of the input row.
        - neu : float
            The neutral sentiment score for the text in the 'line' column of the input row.
        - pos : float
            The positive sentiment score for the text in the 'line' column of the input row.
        - compound : float
            The compound sentiment score for the text in the 'line' column of the input row.
    """
    # Apply sentiment analysis to the 'line' column of the input row using the SentimentIntensityAnalyzer from the nltk library
    sentiment = sia.polarity_scores(row['line'])

    # Return a new pandas Series containing the polarity scores for the input row
    return pd.Series(sentiment)

def analyze_sentiment_tb(row):
    """
    Analyzes the sentiment of a given row using the TextBlob library and returns the resulting sentiment scores as a 
    pandas Series containing polarity and subjectivity.
    This function will be used with the sentiment_analysis function to apply sentiment analysis to every line of the Friends TV show script.

    Parameters:
    row (pandas Series): A row of a pandas DataFrame containing a 'line' column which represents a line of dialogue.

    Returns:
    pandas Series: A pandas Series containing sentiment scores as follows:
        - polarity: A float between -1 and 1 representing the sentiment polarity of the line, where values > 0 indicate 
        positive sentiment, values < 0 indicate negative sentiment, and values close to 0 indicate neutral sentiment.
        - subjectivity: A float between 0 and 1 representing the subjectivity of the line, where values close to 0 indicate 
        objective statements and values close to 1 indicate subjective statements.

    """
    # Apply sentiment analysis to the 'line' column of the input row using the TextBlob library
    sentiment = TextBlob(row['line']).sentiment

    # Return a new pandas Series containing the polarity scores for the input row
    return pd.Series(sentiment)

def sentiment_analysis(friends_script):
    """
    Conducts sentiment analysis on the lines in the Friends script using both the SentimentIntensityAnalyzer 
    and TextBlob. 

    Parameters:
    -----------
    friends_script: pandas.DataFrame
        DataFrame containing the Friends script data

    Returns:
    --------
    pandas.DataFrame
        DataFrame with new columns for the sentiment scores generated by both the SentimentIntensityAnalyzer (SIA) and 
        TextBlob

    """
    # Print status messages to keep the user updated since this process can be very slow.
    print("Starting sentiment analyisis with SIA... this will take a VERY long time")
    
    # Apply the function to each row and create new columns for the sentiment scores using SIA
    friends_script[['sia_neg', 'sia_neu', 'sia_pos', 'sia_compound']] = friends_script.apply(analyze_sentiment_sia, axis=1)
    
    # Print status messages to keep the user updated since this process can be very slow.
    print("Done!\nStarting sentiment analyisis with TextBlob... this one is faster")
    
    # Apply the function to each row and create new columns for the sentiment scores using TextBlob
    friends_script[['tb_polarity', 'tb_subjectivity']] = friends_script.apply(analyze_sentiment_tb, axis=1)
    print("Done!")

    return friends_script
    
def rename_columns_for_sql(friends_script, f_seasons):
    """
    Renames columns in friends_script and f_seasons to make them SQL-friendly.
    
    Args:
    friends_script (DataFrame): DataFrame containing Friends TV show script data.
    f_seasons (DataFrame): DataFrame containing Friends TV show season data.
    
    Returns:
    Tuple: A tuple containing the two DataFrames with renamed columns.
    """
    # Renames columns in friends_script
    friends_script = friends_script.rename(columns={'character': 'f_char', 'line' : 'f_line'})
    
    # Renames columns in f_seasons
    f_seasons = f_seasons.rename(columns={'No.overall' : 'ep_number_overall', 'No. inseason':'ep_number_season', 'Title':'ep_title','Directed by' : "directed_by", "Written by":"written_by", "Original air date":"org_air_date", "Prod.code": "prod_code", "U.S. viewers(millions)" :"us_viewers_mm", "Rating(18–49)":"rating_1", "Rating/share(18–49)" : "rating_2", "Special No." : "special_num" , "U.S. viewersmillions" : "us_viewers_mm_2"})
    return friends_script, f_seasons

def clean_script(friends_script):
    """
    This function takes a Pandas DataFrame containing the Friends script data and performs several data cleaning, 
    preprocessing, and feature engineering steps to prepare it for analysis. 
    
    Parameters:
    -----------
    friends_script : pd.DataFrame
        A Pandas DataFrame containing the Friends script data.
    
    Returns:
    --------
    Tuple[friends_script, f_scene_info, f_seasons]
        A tuple containing three cleaned DataFrames:
        1. The cleaned Friends script DataFrame containing one row per line including the scene it belongs to and the character that spoke that line.
        2. A DataFrame containing information about each scene, description and episode number.
        3. A DataFrame containing information about each episode, including the episode title, air date and season.
    """
    # Clean Friends script
    print("Cleaning Friends script...")
    friends_script = clean_friends_script(friends_script)

    # Extract scene info
    print("Extracting scene info...")
    f_scene_info = create_scene_info(friends_script)

    # Fetch season data from Wikipedia
    print("Fetching season data...")
    f_seasons = get_seasons()

    # Match episode names with seasons
    print("Matching episode names with seasons...")
    f_scene_info, f_seasons = match_episodes(f_scene_info, f_seasons)

    # Match episode names with script
    print("Matching episode names with script...")
    friends_script = match_episode_numbers_in_script(friends_script,f_scene_info)

    # Process character names to standarize them for matching
    print("Processing character names...")
    friends_script = process_character_names(friends_script)
    
    # Perform sentiment analysis using TextBlob and SIA
    friends_script = sentiment_analysis(friends_script)

    # Rename columns for easier processing with SQL
    print("Renaming columns for SQL...")
    friends_script, f_seasons = rename_columns_for_sql(friends_script, f_seasons)
    
    return friends_script, f_scene_info, f_seasons

def export_friends_info_csv(friends_script, f_seasons, f_scene_info):
    """
    Export cleaned Friends script, season info and scene info as csv files

    Parameters:
    friends_script (pandas DataFrame): cleaned Friends script
    f_seasons (pandas DataFrame): Friends TV show seasons data
    f_scene_info (pandas DataFrame): scene information from Friends script

    Returns:
    None
    """
    # Exports every Pandas DataFrame to a CSV using the separator "~" since the text contains several comas and other special characters.
    friends_script.to_csv('data/script.csv', sep='~')
    f_seasons.to_csv('data/seasons.csv', sep='~')
    f_scene_info.to_csv('data/scenes.csv', sep='~')

    print("Done! exported to 'data/' ")
    pass