#!/usr/bin/env python3

"""
Functions:

- Auto checker and notify on huge opportunity (50%+ ROI)
- Bet list
- Bet management (CSV file maker)
- Auto bet through portfolio manager
- Hide taken bets

https://www.hltv.org/betting/money
http://www.aussportsbetting.com/tools/online-calculators/arbitrage-calculator/

"""

import requests
from bs4 import BeautifulSoup
from prettytable import PrettyTable
import re
import time

def calculateOdds(stake, underdog, favourite): # https://github.com/cribbinm/Betting-Arbitrage/blob/master/tools/calculator.py
    underdog_amount = (float(stake) * float(favourite)) / (float(underdog) + float(favourite))
    favourite_amount = (float(stake) * float(underdog)) / (float(underdog) + float(favourite))
    profit = (float(stake) * float(underdog) * float(favourite)) / (float(underdog) + float(favourite)) - float(stake)
    return [round(underdog_amount, 1), round(favourite_amount, 1), round(profit, 1), round(((profit / stake) * 100), 2)]


if __name__ == "__main__" :
    print("\n=== arbitragusMaximus 1.0 by Hartmund Wendlandt ===\n")
    wager = 100 #int(input("Enter your wager in €: "))
    print()

    result = requests.get("https://www.hltv.org/betting/money")
    c = result.content
    soup = BeautifulSoup(c, "lxml")
    bos = soup.find_all(class_="bet-best-of")
    teamrows = soup.find_all("tr", "teamrow")

    team_dict = {} # dictionary for one team
    team_list = []
    bo_list = []
    
    t = PrettyTable(['#', 'Match (Wager: ' + str(wager) + "€)", 'Type', 'ROI (%)', 'Profit (€)', 'Bet 1 (€)', 'Bet 2 (€)'])

    for bo in bos:
        bo_list.append(bo.getText())

    for row in teamrows: # extracts "team - odd - beturl"
        odd_dict = {} # a dictionary of all odds per row so the highest can be sorted out later
        team_dict["name"] = row.find(class_="team-name").get_text()
        team_dict['match_url'] = "https://hltv.org" + row.find("a").get("href")
        
        odds = row.find_all(class_="odds")
        for odd in odds:
            if odd.get_text() and 'hidden' not in odd.attrs['class']: # go through all displayed odds
                odd_url = odd.find('a').get('href')
                odd_value = float(odd.find('a').get_text())
                odd_dict[odd_url] = odd_value

        if odd_dict:    
            team_dict["url"] = max(odd_dict, key=odd_dict.get)
            team_dict["odds"] = max(odd_dict.values())
            team_list.append(team_dict["name"] + "," + str(team_dict["odds"]) + "," + team_dict["url"] + "," + team_dict["match_url"])

    for i in range (0, len(team_list), 2): # for each match
        team1 = team_list[i].split(",")[0]
        team1_odds = team_list[i].split(",")[1]
        team1_url = team_list[i].split(",")[2]
        team2 = team_list[i + 1].split(",")[0]
        team2_odds = team_list[i + 1].split(",")[1]
        team2_url = team_list[i + 1].split(",")[2]
        match_url = team_list[i].split(",")[3]
        bo = bo_list[i]

        result = calculateOdds(wager, team1_odds, team2_odds)
        profit = result[2]
        roi = result[3]
        bet1 = result[0]
        bet2 = result[1]

        if result[3] > 10: # ROI
            # check if match already live
            result = requests.get(match_url) # go to match page
            c = result.content
            soup = BeautifulSoup(c, "lxml")

            unix_timestamp = str(soup.find(class_="event-time"))
            unix_timestamp = re.findall("[0-9]{10}", unix_timestamp)[0]
            
            if int(time.time()) < int(unix_timestamp): # match not yet live
                t.add_row([str(i), team1 + " (" + team1_odds + ")" + " vs " + team2 + " (" + team2_odds + ")", str(bo), str(roi), str(profit), bet1, bet2])
                print("#" + str(i) + " : " + str(match_url) + "\n" + team1 + ": " + str(team1_url) + "\n" + team2 + ": " + str(team2_url) + "\n")

    t.sortby = "Profit (€)"
    t.sort_key = key=lambda k: float(k[4])
    t.reversesort = True
    print(t)