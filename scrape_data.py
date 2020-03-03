"""
Process
1. Surrender all objections and allow the highest vision of this project to manifest
2. Scrape relevant HLTV data (player, team, match)
3. Scrape betting sites data (odds)
4. Statistical analysis magic to calculate most profitable bets
5. print results + best site to bet at + team news (e.g.: https://www.hltv.org/team/6665/astralis#tab-newsBox)

Significant Player Variables (e.g.: https://www.hltv.org/stats/players/4954/Xyp9x?startDate=2019-11-27&endDate=2020-02-27)
* Rating 2.0 (https://www.hltv.org/news/20695/introducing-rating-20)
* Maps played
(* Rating 2.0 vs specific teams)

Significant Team Variables (e.g.: https://www.hltv.org/stats/teams/6665/Astralis?startDate=2019-11-27&endDate=2020-02-27)
* Ranking development for core/roster
* Current win streak
* Win rate last 3 months
* Map win percentage last 3 months
* K/D Ratio

Significant Match Variables (e.g.: https://www.hltv.org/matches/2339394/astralis-vs-fnatic-iem-katowice-2020)
* Map win % for picks
* Live win probability

Program Parameters
* -a to display all bets (without -a, display only bets that should be taken)

Betting sites
* https://gg.bet/
* https://www.bet365.com/
* https://esports.betway.com/
* https://thunderpick.com/
* https://loot4.bet/
* ...
"""

import os
import importlib.util
import pprint
import csv
from datetime import datetime, timedelta
import numpy as np
from sklearn.model_selection import train_test_split

# import hltv api
# spec = importlib.util.spec_from_file_location("hltv", os.path.dirname(os.path.abspath( __file__ )) + "/api/hltv-api/main.py")
# hltv = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(hltv)

def import_csv(csvfilename): # https://stackoverflow.com/a/53483446
    """
    This funciton takes a csv file and returns a array of arrays
    [[    ],
     [    ],
     [    ],
      ...   ]
    """
    data_x = [] # Feature
    data_y = [] # Goldlabel
    row_index = 1
    with open(csvfilename, "r", encoding="utf-8", errors="ignore") as scraped:
        reader = csv.reader(scraped, delimiter=',')
        first_row = next(reader)  # skip to second line, because first doent have values
        for row in reader:
            if row:  # avoid blank lines
                row_index += 1
                x = row[5:-2] # take feature information
                y = float(row[-1]) # take Goldlabel
                x = [0.0 if elem == "-" else float(elem) for elem in x] # remove "-" und set to float
                data_x.append(x)
                data_y.append(y)
        return np.array(data_x), np.array(data_y) # make numpy array


if __name__ == "__main__" :
    pp = pprint.PrettyPrinter()




    data_x, data_y = import_csv("Test 1.csv")
    print(data_x)
    print(data_x.shape)
    print(data_y)
    print(data_y.shape)

    """
    date, event, url, team1, team2, team1_rank, team1_weighted_rank, team2_rank, team2_weighted_rank, team1_player1_rating, team1_player2_rating, team1_player3_rating,
    team1_player4_rating, team1_player5_rating, team1_player1_weighted_rating, team1_player2_weighted_rating, team1_player3_weighted_rating,
    team1_player4_weighted_rating, team1_player5_weighted_rating, team2_player1_rating, team2_player2_rating, team2_player3_rating,
    team2_player4_rating, team2_player5_rating, team2_player1_weighted_rating, team2_player2_weighted_rating, team2_player3_weighted_rating,
    team2_player4_weighted_rating, team2_player5_weighted_rating, winner
    """

    # continue importing matches

    last_match = "" #saved_matches[-1]
    #if last_match[0] >= datetime.now() - timedelta(days=5): # last match older than 5 days
     #   print("Match import complete.")
      #  exit

    new_matches = []
    results = []

    while True: # import matches one month at a time

        if len(saved_matches) < 2 and len(new_matches) == 0: # no matches added yet
            print("Importing matches from 2017-01-01 to 2017-01-02 ...")
            startdate = "2017-01-12"
            enddate = "2017-01-12" # 2020-02-28

        results = hltv.get_results_by_date(startdate, enddate) # yyyy-mm-dd

        for result in results:
            if result["team1score"] > result["team2score"]: winner = result["team1"]
            elif result["team2score"] > result["team1score"]: winner = result["team2"]
            else: winner = "Draw"

            new_matches.append(result["date"] + "," + result["event"] + "," + result["team1"] + "," + result["team2"] + "," + winner)
            print(new_matches[-1])