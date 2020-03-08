"""
This program scrapes all the HLTV data we could EVER need
"""

from bs4 import BeautifulSoup
import requests
from datetime import datetime
import time
import re

DEBUG = 1

def parse_page(url):
    result = requests.get(url)
    c = result.content
    soup = BeautifulSoup(c, "lxml")
    return soup

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

    for i in range (0, matches*2, 2): # for each match
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

        if i < matches:
            if DEBUG == 1:
                print("Sleeping for 10 seconds to prevent IP ban")
            time.sleep(10)

        #all_matches.append(date, event, match_url, team1, team2, ranks_list[:5], weighted_ranks_list[:5], top10_list[:5], maps_played_list[:5], ranks_list[-5:], weighted_ranks_list[-5:], top10_list[-5:], maps_played_list[-5:])
    return all_matches


if __name__ == "__main__" :

    print(import_upcoming_matches(3))