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
nltk.downloader.download('vader_lexicon')
from textblob import TextBlob
pd.set_option('mode.chained_assignment', None)

## Function declaration

def download_friends_script():
    """
    This function downloads the Friends tv show script from the "Friends TV Show Script" from Kaggle using the Kaggle API.
    """
    os.system('kaggle datasets download -d divyansh22/friends-tv-show-script --unzip --p "data/"')
    
    return "Done! Downloaded to data/"


# Define a function to extract the relevant information from each line of the script
def process_line(line):
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
    lines = []
    for line in script:
        lines.append(process_line(line))
    return pd.DataFrame(lines, columns=["episode", "scene", "character", "line"])

def clean_friends_script(friends_script):
    friends_script.replace("", pd.NA, inplace=True)
    friends_script["episode"].fillna(method="ffill", inplace=True)
    friends_script["scene"].fillna(method="ffill", inplace=True)
    friends_script = friends_script.loc[(friends_script['character'] != 'none') & (friends_script['line'].notna())]
    return friends_script

def create_scene_info(friends_script):
    # Scenes per episode
    f_scene_info =friends_script[["episode","scene"]]
    f_scene_info.drop_duplicates(inplace=True)
    f_scene_info.dropna(inplace=True)
    f_scene_info.reset_index(inplace=True)
    f_scene_info.drop("index", axis =1, inplace=True)
    f_scene_info.reset_index(inplace=True)
    f_scene_info.rename(columns={"index":"scene_number"}, inplace=True)
    f_scene_info["episode"] = f_scene_info["episode"].str.title()
    return f_scene_info

def get_seasons():
    seasons_url="https://en.wikipedia.org/wiki/List_of_Friends_episodes"
    seasons_list = pd.read_html(seasons_url)
    f_seasons = []
    for i in list(range(1,12)):
        f_seasons.append(seasons_list[i])

    for i, df in enumerate(f_seasons):
        df["season"] = f'{i+1}'

    f_seasons = pd.concat(f_seasons, ignore_index=True)
    f_seasons["No.overall"] = f_seasons["No.overall"].fillna(-1)
    f_seasons["No.overall"] = f_seasons["No.overall"].astype(int)
    f_seasons["No. inseason"] = f_seasons["No. inseason"].fillna(-1)
    f_seasons["No. inseason"] = f_seasons["No. inseason"].astype(int)
    f_seasons = f_seasons.applymap(lambda x: x if pd.isnull(x) else re.sub(r'\[.*?\]', '', str(x)))
    f_seasons['Title'] = f_seasons['Title'].str.replace('"', '')
    f_seasons.loc[0, 'Title'] = 'The One Where Monica Gets A New Roommate'
    f_seasons["Title"] = f_seasons["Title"].str.title()
    return f_seasons


def match_episodes(f_scene_info, f_seasons):
    # Merge the data frames based on the common column "episode"
    merged_df = pd.merge(f_scene_info, f_seasons[['No.overall', 'Title', 'season']], left_on='episode', right_on='Title', how='left')
    # Replace the "episode" column with the "No.overall" column where it is available
    merged_df['episode'] = merged_df['No.overall'].fillna(merged_df['episode'])
    # Drop unnecessary columns
    merged_df.drop(['No.overall', 'Title'], axis=1, inplace=True)
    # Replace manually non matching episodes
    merged_df['episode'] = merged_df['episode'].replace('The One Where Monica Gets A New Roomate', '1')
    merged_df['episode'] = merged_df['episode'].replace('The One With Two Parts, Part 1', '16')
    merged_df['episode'] = merged_df['episode'].replace('The One With Two Parts, Part 2', '17')
    merged_df['episode'] = merged_df['episode'].replace("The One With Ross' New Girlfriend",'25' )
    merged_df['episode'] = merged_df['episode'].replace('The One Where Mr. Heckles Dies', '27')
    merged_df['episode'] = merged_df['episode'].replace('The One With The Last', '32')
    merged_df['episode'] = merged_df['episode'].replace("The One Where No-One'S Ready", '50')
    merged_df['episode'] = merged_df['episode'].replace('The One Where Monica And Richard Are Friends', '13')
    merged_df['episode'] = merged_df['episode'].replace('The One The Morning After', '64')
    merged_df['episode'] = merged_df['episode'].replace("The One With A Chick. And A Duck", '69')
    merged_df['episode'] = merged_df['episode'].replace("The One Where Chandler Crosses A Line", '80')
    merged_df['episode'] = merged_df['episode'].replace("The One Where They'Re Gonna Party!", '82')
    merged_df['episode'] = merged_df['episode'].replace("The One With Phoebes Uterus", '84')
    merged_df['episode'] = merged_df['episode'].replace("The One With Rachels Crush", '86')
    merged_df['episode'] = merged_df['episode'].replace("The One With Joeys Dirty Day", '87')
    merged_df['episode'] = merged_df['episode'].replace("The One With Ross'S Wedding Parts I And Ii", '96')
    merged_df['episode'] = merged_df['episode'].replace("The One With Ross'S Wedding - Uncut Version", '97')
    merged_df['episode'] = merged_df['episode'].replace("The One Hundredth", '100')
    merged_df['episode'] = merged_df['episode'].replace("The One With All The Kips", '102')
    merged_df['episode'] = merged_df['episode'].replace("The One With The Thanksgiving Flashbacks", '105')
    merged_df['episode'] = merged_df['episode'].replace("The One Where Everyone Finds Out", '111')
    merged_df['episode'] = merged_df['episode'].replace("The One With A Cop", '113')
    merged_df['episode'] = merged_df['episode'].replace("The One With Rachel'S Inadvertant Kiss", '114')
    merged_df['episode'] = merged_df['episode'].replace("The One With The Ride Along", '117')
    merged_df['episode'] = merged_df['episode'].replace("The One With Rosss Denial", '124')
    merged_df['episode'] = merged_df['episode'].replace("The One With Joeys Porsche", '126')
    merged_df['episode'] = merged_df['episode'].replace("The One The Last Night", '127')
    merged_df['episode'] = merged_df['episode'].replace("The One With Rachels Sister", '134')
    merged_df['episode'] = merged_df['episode'].replace("The One Where Chandler Cant Cry", '135')
    merged_df['episode'] = merged_df['episode'].replace("The One With The Unagi", '138')
    merged_df['episode'] = merged_df['episode'].replace("The One With Joeys Fridge", '140')
    merged_df['episode'] = merged_df['episode'].replace("The One Where Ross Meets Elizabeths Dad", '142')
    merged_df['episode'] = merged_df['episode'].replace("The One Where Pauls The Man", '143')
    merged_df['episode'] = merged_df['episode'].replace("The One With Monicas Thunder", '147')
    merged_df['episode'] = merged_df['episode'].replace("The One With Rachels Book", '148')
    merged_df['episode'] = merged_df['episode'].replace("The One With Phoebes Cookies", '149')
    merged_df['episode'] = merged_df['episode'].replace("The One With Rachels Assistant", '150')
    merged_df['episode'] = merged_df['episode'].replace("The One With Rosss Book", '153')
    merged_df['episode'] = merged_df['episode'].replace("The One Where Chandler Doesnt Like Dogs", '154')
    merged_df['episode'] = merged_df['episode'].replace("The One With All The Cheesecake", '157')
    merged_df['episode'] = merged_df['episode'].replace("The One Where Theyre Up All Night", '158')
    merged_df['episode'] = merged_df['episode'].replace("The One Where Rosita Dies", '159')
    merged_df['episode'] = merged_df['episode'].replace("The One Where They All Turn Thirty", '160')
    merged_df['episode'] = merged_df['episode'].replace("The One With Joeys New Brain", '161')
    merged_df['episode'] = merged_df['episode'].replace("The One With The Truth About London", '162')
    merged_df['episode'] = merged_df['episode'].replace("The One With Joeys Award", '164')
    merged_df['episode'] = merged_df['episode'].replace("The One With Ross And Monicas Cousin", '165')
    merged_df['episode'] = merged_df['episode'].replace("The One With Rachels Big Kiss", '166')
    merged_df['episode'] = merged_df['episode'].replace("The One With Chandlers Dad", '168')
    merged_df['episode'] = merged_df['episode'].replace("The One With Chandler And Monicas Wedding", '169')
    merged_df['episode'] = merged_df['episode'].replace('The One After "I Do"', '171')
    merged_df['episode'] = merged_df['episode'].replace("The One With Monicas Boots", '180')
    merged_df['episode'] = merged_df['episode'].replace("The One With Ross' Big Step Forward", '181')
    merged_df['episode'] = merged_df['episode'].replace("The One In Massapequa", '188')
    merged_df['episode'] = merged_df['episode'].replace("The One With Joeys Interview", '189')
    # now lets re- match the seasons
    # Merge the data frames based on the common column "episode"
    merged_df.drop(['season'], axis=1, inplace=True)
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
    merged_df.drop(['season'], axis=1, inplace=True)
    merged_df = pd.merge(merged_df, f_seasons[['No.overall', 'season']], left_on='episode', right_on='No.overall', how='left')
    f_scene_info = merged_df
    # I will drop no.overall and season since this information is already in our f_easons df
    f_scene_info.drop(["season", "No.overall_x", "No.overall_y"], axis=1, inplace=True)
    return f_scene_info, f_seasons


def match_episode_numbers_in_script(friends_script,f_scene_info):
    # now we need to add the episode number to our script so that we can relate all 3 tables. The writer info is redundant so lets start by dropping thos rows:
    friends_script = friends_script.dropna(subset=['scene'])
    friends_script['episode'] = friends_script['episode'].str.title()
    merged_df_scenes = pd.merge(friends_script, f_scene_info[['scene_number', 'episode','scene']], left_on='scene', right_on='scene', how='left')
    merged_df_scenes.drop(["episode_x", "scene", "episode_y"], axis=1, inplace=True)
    friends_script = merged_df_scenes
    return friends_script

def process_character_names(friends_script):
    # There are a lot of character names that are formatted differently or that include actions. Lets replace those:
    friends_script["character"] = friends_script["character"].str.replace(r"^(Ross|Rachel|Monica|Joey|Chandler|Phoebe) \(.+", r"\1", regex=True)
    friends_script["character"] = friends_script["character"].str.replace(r"^(Ross|Rachel|Monica|Joey|Chandler|Phoebe) $", r"\1", regex=True)
    # Now lets replace all names spelled in uppercase
    uppercase_names = {"MONICA": "Monica", "CHANDLER": "Chandler", "JOEY": "Joey", "PHOEBE": "Phoebe", "RACHELL": "Rachel", "ROSS": "Ross"}
    friends_script["character"] = friends_script["character"].replace(uppercase_names, regex=True)

    # Finally lets replace abbreviated names:
    abrev_names = {"MNCA": "Monica", "CHAN": "Chandler", "JOEY": "Joey", "PHOE": "Phoebe", "RACH": "Rachel", "ROSS": "Ross"}
    friends_script["character"] = friends_script["character"].replace(abrev_names, regex=True)

    return friends_script

# I will initialize the sentiment analizer outside the function as this function is used in an apply and will slow down the process considerably.
sia = SentimentIntensityAnalyzer()

def analyze_sentiment_sia(row):
    sentiment = sia.polarity_scores(row['line'])
    return pd.Series(sentiment)

def analyze_sentiment_tb(row):
    sentiment = TextBlob(row['line']).sentiment
    return pd.Series(sentiment)

def sentiment_analysis(friends_script):
    # apply the function to each row and create new columns for the sentiment scores
    print("Starting sentiment analyisis with SIA... this will take a VERY long time")
    sia = SentimentIntensityAnalyzer()
    friends_script[['sia_neg', 'sia_neu', 'sia_pos', 'sia_compound']] = friends_script.apply(analyze_sentiment_sia, axis=1)
    print("Done! \n Starting sentiment analyisis with TextBlob... this one is faster")
    friends_script[['tb_polarity', 'tb_subjectivity']] = friends_script.apply(analyze_sentiment_tb, axis=1)
    print("Done!")
    return friends_script
    
def rename_columns_for_sql(friends_script, f_seasons):
    friends_script = friends_script.rename(columns={'character': 'f_char', 'line' : 'f_line'})
    f_seasons = f_seasons.rename(columns={'No.overall' : 'ep_number_overall', 'No. inseason':'ep_number_season', 'Title':'ep_title','Directed by' : "directed_by", "Written by":"written_by", "Original air date":"org_air_date", "Prod.code": "prod_code", "U.S. viewers(millions)" :"us_viewers_mm", "Rating(18–49)":"rating_1", "Rating/share(18–49)" : "rating_2", "Special No." : "special_num" , "U.S. viewersmillions" : "us_viewers_mm_2"})
    return friends_script, f_seasons

def clean_script(friends_script):
    print("Cleaning Friends script...")
    friends_script = clean_friends_script(friends_script)
    print("Extracting scene info...")
    f_scene_info = create_scene_info(friends_script)
    print("Fetching season data...")
    f_seasons = get_seasons()
    print("Matching episode names with seasons...")
    f_scene_info, f_seasons = match_episodes(f_scene_info, f_seasons)
    print("Matching episode names with script...")
    friends_script = match_episode_numbers_in_script(friends_script,f_scene_info)
    print("Processing character names...")
    friends_script = process_character_names(friends_script)
    
    friends_script = sentiment_analysis(friends_script)
    print("Renaming columns for SQL...")
    friends_script, f_seasons = rename_columns_for_sql(friends_script, f_seasons)
    
    return friends_script, f_scene_info, f_seasons


def export_friends_info_csv(friends_script, f_seasons, f_scene_info):
    friends_script.to_csv('data/script.csv', sep='~')
    f_seasons.to_csv('data/seasons.csv', sep='~')
    f_scene_info.to_csv('data/scenes.csv', sep='~')
    print("Done! exported to data/")
    pass
