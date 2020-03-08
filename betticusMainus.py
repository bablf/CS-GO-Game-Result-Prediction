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
import requests
import re
from datetime import datetime
import time
from bs4 import BeautifulSoup
import keras
from sklearn.model_selection import train_test_split

DEBUG = 1

def parse_page(url):
    result = requests.get(url)
    c = result.content
    soup = BeautifulSoup(c, "lxml")
    if "rate limited" or "CAPTCHA" in soup:
        print("\n\nHLTV block on parsing, exiting.\n\n")
        raise SystemExit
    return soup

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

def import_upcoming_matches(matches):
    """
    This function parses the upcoming matches from https://www.hltv.org/betting/money ,
    parses the match pages, parses the player pages, and returns an array of arrays

    [
        [ # match
        [] # row like in csv
        ],
    ]

    """

    soup = parse_page("https://www.hltv.org/betting/money")
    teamrows = soup.find_all("tr", "teamrow")
    team_dict = {} # dictionary for one team
    team_list = []
    all_matches = [] # master list that is returned for NN

    for row in teamrows: # extracts "team - odd - beturl"
        team_dict["name"] = row.find(class_="team-name").get_text()
        team_dict['match_url'] = "https://hltv.org" + row.find("a").get("href").replace("/betting/analytics", "/matches")
        team_list.append(team_dict["name"] + "," + team_dict["match_url"])

    if DEBUG == 1:
        print(str(len(team_list)) + " teams scrapped from https://hltv.org/betting/money. Importing " + str(matches) + " matches due to parameters.\n")

    if matches == -1:
        matches = len(team_list) # scrape all matches

    for i in range (0, matches, 2): # for each match
        team1 = team_list[i].split(",")[0]
        team2 = team_list[i + 1].split(",")[0]
        match_url = team_list[i].split(",")[1]

        soup = parse_page(match_url) # go to match page
        unix_timestamp = str(soup.find(class_="timeAndEvent"))
        unix_timestamp = re.findall("[0-9]{10}", unix_timestamp)[0]

        if int(time.time()) > int(unix_timestamp):
            if DEBUG == 1:
                print("Match already live or closed, skipping.\n")
            continue # skip past / live matches

        date = datetime.utcfromtimestamp(int(unix_timestamp)).strftime('X%m/X%d/%Y').replace('X0', 'X').replace('X','') # mm/dd/yyyy
        event = str(soup.find(class_="event text-ellipsis").get_text())

        # extract 3 month rating for all players
        container = str(soup.find(class_="lineups-compare-container"))
        weighted_ranks_list = re.findall('\"rating\":\"[0-2].[0-9]{2}\"', container) # ['"rating":"1.02"', '"rating":"0.94"', '"rating":"0.98"', '"rating":"1.05"', '"rating":"1.09"', '"rating":"1.04"', '"rating":"1.10"', '"rating":"0.94"', '"rating":"1.01"', '"rating":"1.18"']
        weighted_ranks_list = re.findall('[0-2].[0-9]{2}', str(weighted_ranks_list)) # ['1.08', '0.99', '0.96', '1.09', '0.84', '1.04', '1.16', '1.12', '1.10', '0.90']

        if DEBUG == 1:
            print("Parsed match page of " + team1 + " vs " + team2 + ".\nTimestamp: " + unix_timestamp + "\nURL: " + match_url + "\n3 month rating of players: " + str(weighted_ranks_list) + "\n")

        # extract player profile links for rank, top10, maps_played
        player_bare_list = re.findall('/stats/players/[0-9]+/\w+', container) # "statsLinkUrl":"/stats/players/10784/RuStY"
        player_link_list = []

        for player_link in player_bare_list: # convert to full links + extract 10 players
            if len(player_link_list) < 10:
                player_link_list.append("https://hltv.org" + player_link + "?startDate=2017-01-01&endDate=" + datetime.now().strftime('%Y-%m-%d'))

        for player_profile in player_link_list: # go to each player page
            soup = parse_page(player_profile)
            stats_row = str(soup.find_all(class_="stats-row"))

            # extract rating
            ranks_list = str(re.findall("Rating 2\.0<\/span><span class=\"strong\">[0-2].[0-9]{2}<\/span><\/div>", stats_row)).replace('Rating 2.0</span><span class=\"strong\">', "").replace("</span></div>", "")

            # extract top10 rating
            opponent_rating = str(soup.find_all(class_="rating-value")[1]).replace('<div class="rating-value">', "").replace("</div>","")
            if opponent_rating == "-":
                opponent_rating = ""

#           top10_list = str(re.findall("", stats_row).replace("", "").replace("", ""))

            # extract maps played
            maps_played_list = str(re.findall("Maps played<\/span><span>[0-9]+<\/span>", stats_row)).replace("Maps played</span><span>","").replace("</span>","")

            if DEBUG == 1:
                print("Parsed player page " + player_profile + "\nRating: " + ranks_list + "\nRating vs Top 10: " + opponent_rating + "\nMaps played: " + maps_played_list + "\n")

        if DEBUG == 1:
            print("Sleeping for 30 seconds to prevent IP ban")
        time.sleep(30)

        #all_matches.append(date, event, match_url, team1, team2, ranks_list[:5], weighted_ranks_list[:5], top10_list[:5], maps_played_list[:5], ranks_list[-5:], weighted_ranks_list[-5:], top10_list[-5:], maps_played_list[-5:])
    return all_matches

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
    """
    data_x, data_y = import_past_matches("/content/drive/My Drive/Colab Notebooks/EzBetticus/matches.csv")
    print(data_x)
    print(data_x.shape)
    print(data_y)
    print(data_y.shape)
    """

    print(import_upcoming_matches(1))