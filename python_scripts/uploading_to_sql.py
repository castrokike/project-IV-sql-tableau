import sqlalchemy as alch
import pymysql
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()


password = os.getenv('SQL_PASSWORD')
dbName = "friends"
connectionData=f"mysql+pymysql://root:{password}@localhost/{dbName}"
engine = alch.create_engine(connectionData)

# Reading the processed information from the CSV files exported in the previous script.
friends_script = pd.read_csv('data/script.csv', sep='~')
f_scene_info = pd.read_csv('data/scenes.csv', sep='~')
f_seasons = pd.read_csv('data/seasons.csv', sep='~')

friends_script.to_sql('script', con=engine, if_exists='replace')
f_scene_info.to_sql('scenes', con=engine, if_exists='replace')
f_seasons.to_sql('seasons', con=engine, if_exists='replace')


