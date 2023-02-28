# project-IV-sql-tableau
## Project IV SQL - Tableau
## Andrés Castro

# Friends Script Analysis
This project analyzes the script of the TV show Friends using Python. The data is obtained from Kaggle and consists of a .txt file containing the full script of the TV Show. The file is processed to extract each line spoken by each character on every episode and obtains information about the season each episode belongs to along with every scene contained in every episode. Finally this information is processed to analyze the sentiment of every line. This information is then graphed using tabluea.

### The project performs the following tasks:

- Cleaning the script data
- Extracting scene information
- Matching episode names between season information extracted from Wikipedia and episode names in the script
- Processing and standarizing character names
- Performing sentiment analysis on the script using two different libraries (TextBlob and SentimentIntensityAnalyzer)
- Exporting the cleaned and analyzed data to CSV files
- Graphing using Tableau to gain insights

## Requirements
The project requires Python 3 and the following libraries:

- pandas
- numpy
- re
- textblob
- nltk
- vaderSentiment

## Usage
The script outputs three CSV files containing the cleaned and analyzed data:

script.csv: Contains the lines spoken by each character in each scene, along with the sentiment scores.
seasons.csv: Contains metadata for each episode, such as the episode number, title, air date, ratings and season it belongs to.
scenes.csv: Contains information about each scene, such as the episode number, scene number, and description of the scene.

## Results
