"""
**Backend**

1. Visit HLTV to scrape matches within x days + corresponding player info (fetch_upcoming_matches)
2. Feed info to NN and return win-probability if > 95% (calculate_winner)
3. Calculate bet size and print gathered info (calculate_betsize)

**Frontend**

1. Ask for total bet budget (to calculate bet size)
2. Show suggested bets with potential profit on website (highlight best site)

**ezBetticus v0.2 ideas**
- Ask if NN should be updated with new matches ("last match is xx vs xx at xx.xx.xxxx")
- Run as daemon that sends email to us when new profitable bets are available
- Create a GUI
- Add to GUI a tab for bet management. Track made bets + export them as CSV (for taxes). P/L calculation, future profit estimation (with graphs), data encryption.
"""

import csv
import numpy as np
import keras
from sklearn.model_selection import train_test_split

DEBUG = 1

def import_past_matches(csvfilename):
  
    """
    This function takes a csv file and returns an array of arrays

    [
        [],
        [],
        [],
    ]

    date, event, url, team1, team2,

    team1_player1_rating, team1_player2_rating, team1_player3_rating, team1_player4_rating, team1_player5_rating,
    team1_player1_weighted_rating, team1_player2_weighted_rating, team1_player3_weighted_rating, team1_player4_weighted_rating, team1_player5_weighted_rating,
    team1_player1_rating_top10, team1_player2_rating_top10, team1_player3_rating_top10, team1_player4_rating_top10, team1_player5_rating_top10, 
    team1_player1_maps_played, team1_player2_maps_played, team1_player3_maps_played, team1_player4_maps_played, team1_player5_maps_played, 

    team2_player1_rating, team2_player2_rating, team2_player3_rating, team2_player4_rating, team2_player5_rating,
    team2_player1_weighted_rating, team2_player2_weighted_rating, team2_player3_weighted_rating, team2_player4_weighted_rating, team2_player5_weighted_rating,
    team2_player1_rating_top10, team2_player2_rating_top10, team2_player3_rating_top10, team2_player4_rating_top10, team2_player5_rating_top10,
    team2_player1_maps_played, team2_player2_maps_played, team2_player3_maps_played, team2_player4_maps_played, team2_player5_maps_played, 

    winner

    """

    data_x = [] # Features
    data_y = [] # Goldlabel
    row_index = 1
    with open(csvfilename, "r", encoding="utf-8", errors="ignore") as scraped:
        reader = csv.reader(scraped, delimiter=',')
        first_row = next(reader)  # skip to second line, because first doesn't have values
        for row in reader:
            if row:  # avoid blank lines
                row_index += 1
                x = row[5:-2] # take feature information
                y = float(row[-1]) # take Goldlabel
                x = [0.0 if elem == "" else float(elem) for elem in x] # set blanks to float
                data_x.append(x)
                data_y.append(y)
        return np.array(data_x), np.array(data_y) # make numpy array

def calculate_winner(match):
  """
  This function uses the array returned from fetch_upcoming_matches, feeds it to the NN, and returns an array

    [winner, probability]

  """

  pass

def calculate_betsize(winning_probability, total_budget):
  """
  This function calculates the bet size and returns an integer


  """

  # round final bet size because betting sites tend to ban users using odd numbers (to fight arbitrage betting)
  pass

if __name__ == "__main__" :

    data_x, data_y = import_past_matches("./data/past_matches.csv")
    print(data_x)
    print(data_x.shape)
    print(data_y)
    print(data_y.shape)