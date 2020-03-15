"""
This program scrapes the best HLTV data EUWEST
"""


from bs4 import BeautifulSoup
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time
import re
import pandas as pd
import sys
import pprint

DEBUG = 1
SCRAPE_TIMEOUT = 0.035

def parsePage(url):
    """
    This function parses an URL and returns its soup
    """
    time.sleep(SCRAPE_TIMEOUT)
    result = requests.get(url)
    return BeautifulSoup(result.content, "lxml")


def parsePlayerProfile(url, startDate, endDate):
    """
    This function returns all player stats. URL example: /10784/RuStY
    """
    
    try:
        soup = parsePage("https://hltv.org/stats/players/" + url + "?startDate=" + startDate + "&endDate=" + endDate) # main tab
        stat_box = str(soup.find_all(class_="summaryStatBreakdownDataValue")).replace('<div class="summaryStatBreakdownDataValue">', '').replace('</div>', '').split(", ")
        stats_row = str(soup.find_all(class_="stats-row"))

        rating = stat_box[0].replace("[", "")
        dpr = stat_box[1]
        kast = stat_box[2].replace("%", "")
        impact = stat_box[3]
        adr = stat_box[4]
        kpr = stat_box[5].replace("]", "")
        kills = str(re.findall("Total kills</span><span>([0-9]+)", stats_row)[0])
        hs = str(re.findall("Headshot %</span><span>(\d+\.\d+)", stats_row)[0])
        deaths = str(re.findall("Total deaths</span><span>([0-9]+)", stats_row)[0])
        apr = str(re.findall("Assists / round</span><span>(\d+\.\d+)", stats_row)[0])
        kd = str(re.findall("K/D Ratio</span><span>(\d+\.\d+)", stats_row)[0])
        dr = str(re.findall("Deaths / round</span><span>(\d+\.\d+)", stats_row)[0])
        sbtr = str(re.findall("Saved by teammate / round</span><span>(\d+\.\d+)", stats_row)[0])
        stpr = str(re.findall("Saved teammates / round</span><span>(\d+\.\d+)", stats_row)[0])
        gdmgr = str(re.findall("Grenade dmg / Round</span><span>(\d+\.\d+)", stats_row)[0]).replace("Grenade dmg / Round</span><span>", "")
        maps = str(re.findall("Maps played</span><span>([0-9]+)", stats_row)[0])
        top5 = str(soup.find_all(class_="rating-value")[0]).replace('<div class="rating-value">', "").replace("</div>","").replace("-", "0")
        top10 = str(soup.find_all(class_="rating-value")[1]).replace('<div class="rating-value">', "").replace("</div>","").replace("-", "0")
        top20 = str(soup.find_all(class_="rating-value")[2]).replace('<div class="rating-value">', "").replace("</div>","").replace("-", "0")
        top30 = str(soup.find_all(class_="rating-value")[3]).replace('<div class="rating-value">', "").replace("</div>","").replace("-", "0")
        top50 = str(soup.find_all(class_="rating-value")[4]).replace('<div class="rating-value">', "").replace("</div>","").replace("-", "0") 

        soup = parsePage("https://hltv.org/stats/players/individual/" + url + "?startDate=" + startDate + "&endDate=" + endDate) # individual tab
        stats_row = str(soup.find_all(class_="stats-row"))

        kpd = str(re.findall("Kill / Death</span><span>(\d+\.\d+)", stats_row)[0]).replace("Kill / Death</span><span>", "")
        rwk = str(re.findall("Rounds with kills</span><span>([0-9]+)", stats_row)[0])
        kdd = str(re.findall('K - D diff.</span><span>(-?[0-9]+)', stats_row)[0])
        kr0 = str(re.findall("0 kill rounds</span><span>([0-9]+)", stats_row)[0])
        kr1 = str(re.findall("1 kill rounds</span><span>([0-9]+)", stats_row)[0])
        kr2 = str(re.findall("2 kill rounds</span><span>([0-9]+)", stats_row)[0])
        kr3 = str(re.findall("3 kill rounds</span><span>([0-9]+)", stats_row)[0])
        kr4 = str(re.findall("4 kill rounds</span><span>([0-9]+)", stats_row)[0])
        kr5 = str(re.findall("5 kill rounds</span><span>([0-9]+)", stats_row)[0])
        tok = str(re.findall("Total opening kills</span><span>([0-9]+)", stats_row)[0])
        tod = str(re.findall("Total opening deaths</span><span>([0-9]+)", stats_row)[0])
        okr = str(re.findall("Opening kill ratio</span><span>(\d+\.\d+)", stats_row)[0])
        okra = str(re.findall("Opening kill rating</span><span>(\d+\.\d+)", stats_row)[0])
        twp = str(re.findall("Team win percent after first kill</span><span>(\d+\.\d+)", stats_row)[0])
        fkwon = str(re.findall("First kill in won rounds</span><span>(\d+\.\d+)", stats_row)[0])
        riflek = str(re.findall("Rifle kills</span><span>([0-9]+)", stats_row)[0])
        sniperk = str(re.findall("Sniper kills</span><span>([0-9]+)", stats_row)[0])
        smgk = str(re.findall("SMG kills</span><span>([0-9]+)", stats_row)[0])
        pistolk = str(re.findall("Pistol kills</span><span>([0-9]+)", stats_row)[0])
        grenadek = str(re.findall("Grenade</span><span>([0-9]+)", stats_row)[0])
        otherk = re.findall("Other</span><span>([0-9]+)", stats_row)[0]
        
        final_list = [rating, dpr, kast, impact, adr, kpr, kills, hs, deaths, apr, kd, dr, sbtr, stpr, gdmgr, maps, top5, top10, top20, top30, top50, kpd, rwk, kdd, kr0, kr1, kr2, kr3, kr4, kr5, tok, tod, okr, okra, twp, fkwon, riflek, sniperk, smgk, pistolk, grenadek, otherk]
    
    except:
        return
    
    if DEBUG == 1:
        print("\nPlayer " + url + " (since " + startDate + "): " + str(final_list))
        
    return final_list

    
def parseUpcomingMatches(matches):
    """
    This function parses the upcoming matches from https://www.hltv.org/betting/money
    
    Return: [date, event, match_url, team1, team2, player1_url, player2_url, player3_url, player4_url, player5_url, player6_url, player7_url, player8_url, player9_url, player10_url]
    
    Use -1 to parse all upcoming matches
    
    """
    
    soup = parsePage("https://www.hltv.org/betting/money")
    match_soup = soup.find_all(class_="bet-container")
    parsed_matches = [] # "url - team1 - team2" for all matches on the page
    match_list = [] # master list
    
    if matches == -1:
        matches = int(len(match_soup))

    if DEBUG == 1:
        print(str(len(match_soup)) + " matches found on https://hltv.org/betting/money. Importing " + str(matches) + " matches due to parameters.\n")

    for match in match_soup:
        parsed_matches.append(["https://hltv.org" + match.find("a").get("href").replace("/betting/analytics", "/matches"), match.find_all(class_="team-name")[0].get_text(), match.find_all(class_="team-name")[1].get_text()])

    for match in parsed_matches[0:matches]: # for each match respecting parameters
        soup = parsePage(match[0]) # parse match page for player urls
        unix_timestamp = str(soup.find(class_="timeAndEvent"))
        unix_timestamp = re.findall("[0-9]{10}", unix_timestamp)[0]

        if int(time.time()) > int(unix_timestamp):
            if DEBUG == 1:
                print("Match already live or closed, skipping.\n")
            continue # skip past / live matches

        date = datetime.utcfromtimestamp(int(unix_timestamp)).strftime('%Y-%m-%d')
        event = str(soup.find(class_="event text-ellipsis").get_text())

        # extract player profile links
        player_bare_list = re.findall(r'/stats/players/[0-9]+/\w+', str(soup)) # "statsLinkUrl":"/stats/players/10784/RuStY"
        player_link_list = []

        for player_link in player_bare_list: # extract 10 players
            if len(player_link_list) < 10:
                player_link_list.append(player_link.replace("/stats/players/", ""))

        match_list.append([date, event, match[0], match[1], match[2], player_link_list])
        
    #if DEBUG == 1:
    #    print(match_list)
        
    return match_list


if __name__ == "__main__":
    
    pp = pprint.PrettyPrinter()
    print("\n=== scraperinusTotalus 1.0 by Hartmund Wendlandt ===\n")
    print("Please select task:\n[1] Scrape upcoming matches\n[2] Scrape past matches\n")
    task = 1 #int(input())
    
    if task == 1:
        # read csv
        # delete old entries
        # get starting date for oldest upcoming match
        
        csv_headers = ['date', 'event', 'url', 'team1', 'team2', 'team1_player1_rating', 'team1_player1_dpr', 'team1_player1_kast', 'team1_player1_impact', 'team1_player1_adr', 'team1_player1_kpr', 'team1_player1_kills', 'team1_player1_hs', 'team1_player1_deaths', 'team1_player1_apr', 'team1_player1_kd', 'team1_player1_dr', 'team1_player1_sbtr', 'team1_player1_stpr', 'team1_player1_gdmgr', 'team1_player1_maps', 'team1_player1_top5', 'team1_player1_top10', 'team1_player1_top20', 'team1_player1_top30', 'team1_player1_top50', 'team1_player1_kpd', 'team1_player1_rwk', 'team1_player1_kdd', 'team1_player1_kr0', 'team1_player1_kr1', 'team1_player1_kr2', 'team1_player1_kr3', 'team1_player1_kr4', 'team1_player1_kr5', 'team1_player1_tok', 'team1_player1_tod', 'team1_player1_okr', 'team1_player1_okra', 'team1_player1_twp', 'team1_player1_fkwon', 'team1_player1_riflek', 'team1_player1_sniperk', 'team1_player1_smgk', 'team1_player1_pistolk', 'team1_player1_grenadek', 'team1_player1_otherk', 'team1_player2_rating', 'team1_player2_dpr', 'team1_player2_kast', 'team1_player2_impact', 'team1_player2_adr', 'team1_player2_kpr', 'team1_player2_kills', 'team1_player2_hs', 'team1_player2_deaths', 'team1_player2_apr', 'team1_player2_kd', 'team1_player2_dr', 'team1_player2_sbtr', 'team1_player2_stpr', 'team1_player2_gdmgr', 'team1_player2_maps', 'team1_player2_top5', 'team1_player2_top10', 'team1_player2_top20', 'team1_player2_top30', 'team1_player2_top50', 'team1_player2_kpd', 'team1_player2_rwk', 'team1_player2_kdd', 'team1_player2_kr0', 'team1_player2_kr1', 'team1_player2_kr2', 'team1_player2_kr3', 'team1_player2_kr4', 'team1_player2_kr5', 'team1_player2_tok', 'team1_player2_tod', 'team1_player2_okr', 'team1_player2_okra', 'team1_player2_twp', 'team1_player2_fkwon', 'team1_player2_riflek', 'team1_player2_sniperk', 'team1_player2_smgk', 'team1_player2_pistolk', 'team1_player2_grenadek', 'team1_player2_otherk', 'team1_player3_rating', 'team1_player3_dpr', 'team1_player3_kast', 'team1_player3_impact', 'team1_player3_adr', 'team1_player3_kpr', 'team1_player3_kills', 'team1_player3_hs', 'team1_player3_deaths', 'team1_player3_apr', 'team1_player3_kd', 'team1_player3_dr', 'team1_player3_sbtr', 'team1_player3_stpr', 'team1_player3_gdmgr', 'team1_player3_maps', 'team1_player3_top5', 'team1_player3_top10', 'team1_player3_top20', 'team1_player3_top30', 'team1_player3_top50', 'team1_player3_kpd', 'team1_player3_rwk', 'team1_player3_kdd', 'team1_player3_kr0', 'team1_player3_kr1', 'team1_player3_kr2', 'team1_player3_kr3', 'team1_player3_kr4', 'team1_player3_kr5', 'team1_player3_tok', 'team1_player3_tod', 'team1_player3_okr', 'team1_player3_okra', 'team1_player3_twp', 'team1_player3_fkwon', 'team1_player3_riflek', 'team1_player3_sniperk', 'team1_player3_smgk', 'team1_player3_pistolk', 'team1_player3_grenadek', 'team1_player3_otherk', 'team1_player4_rating', 'team1_player4_dpr', 'team1_player4_kast', 'team1_player4_impact', 'team1_player4_adr', 'team1_player4_kpr', 'team1_player4_kills', 'team1_player4_hs', 'team1_player4_deaths', 'team1_player4_apr', 'team1_player4_kd', 'team1_player4_dr', 'team1_player4_sbtr', 'team1_player4_stpr', 'team1_player4_gdmgr', 'team1_player4_maps', 'team1_player4_top5', 'team1_player4_top10', 'team1_player4_top20', 'team1_player4_top30', 'team1_player4_top50', 'team1_player4_kpd', 'team1_player4_rwk', 'team1_player4_kdd', 'team1_player4_kr0', 'team1_player4_kr1', 'team1_player4_kr2', 'team1_player4_kr3', 'team1_player4_kr4', 'team1_player4_kr5', 'team1_player4_tok', 'team1_player4_tod', 'team1_player4_okr', 'team1_player4_okra', 'team1_player4_twp', 'team1_player4_fkwon', 'team1_player4_riflek', 'team1_player4_sniperk', 'team1_player4_smgk', 'team1_player4_pistolk', 'team1_player4_grenadek', 'team1_player4_otherk', 'team1_player5_rating', 'team1_player5_dpr', 'team1_player5_kast', 'team1_player5_impact', 'team1_player5_adr', 'team1_player5_kpr', 'team1_player5_kills', 'team1_player5_hs', 'team1_player5_deaths', 'team1_player5_apr', 'team1_player5_kd', 'team1_player5_dr', 'team1_player5_sbtr', 'team1_player5_stpr', 'team1_player5_gdmgr', 'team1_player5_maps', 'team1_player5_top5', 'team1_player5_top10', 'team1_player5_top20', 'team1_player5_top30', 'team1_player5_top50', 'team1_player5_kpd', 'team1_player5_rwk', 'team1_player5_kdd', 'team1_player5_kr0', 'team1_player5_kr1', 'team1_player5_kr2', 'team1_player5_kr3', 'team1_player5_kr4', 'team1_player5_kr5', 'team1_player5_tok', 'team1_player5_tod', 'team1_player5_okr', 'team1_player5_okra', 'team1_player5_twp', 'team1_player5_fkwon', 'team1_player5_riflek', 'team1_player5_sniperk', 'team1_player5_smgk', 'team1_player5_pistolk', 'team1_player5_grenadek', 'team1_player5_otherk', 'team2_player1_rating', 'team2_player1_dpr', 'team2_player1_kast', 'team2_player1_impact', 'team2_player1_adr', 'team2_player1_kpr', 'team2_player1_kills', 'team2_player1_hs', 'team2_player1_deaths', 'team2_player1_apr', 'team2_player1_kd', 'team2_player1_dr', 'team2_player1_sbtr', 'team2_player1_stpr', 'team2_player1_gdmgr', 'team2_player1_maps', 'team2_player1_top5', 'team2_player1_top10', 'team2_player1_top20', 'team2_player1_top30', 'team2_player1_top50', 'team2_player1_kpd', 'team2_player1_rwk', 'team2_player1_kdd', 'team2_player1_kr0', 'team2_player1_kr1', 'team2_player1_kr2', 'team2_player1_kr3', 'team2_player1_kr4', 'team2_player1_kr5', 'team2_player1_tok', 'team2_player1_tod', 'team2_player1_okr', 'team2_player1_okra', 'team2_player1_twp', 'team2_player1_fkwon', 'team2_player1_riflek', 'team2_player1_sniperk', 'team2_player1_smgk', 'team2_player1_pistolk', 'team2_player1_grenadek', 'team2_player1_otherk', 'team2_player2_rating', 'team2_player2_dpr', 'team2_player2_kast', 'team2_player2_impact', 'team2_player2_adr', 'team2_player2_kpr', 'team2_player2_kills', 'team2_player2_hs', 'team2_player2_deaths', 'team2_player2_apr', 'team2_player2_kd', 'team2_player2_dr', 'team2_player2_sbtr', 'team2_player2_stpr', 'team2_player2_gdmgr', 'team2_player2_maps', 'team2_player2_top5', 'team2_player2_top10', 'team2_player2_top20', 'team2_player2_top30', 'team2_player2_top50', 'team2_player2_kpd', 'team2_player2_rwk', 'team2_player2_kdd', 'team2_player2_kr0', 'team2_player2_kr1', 'team2_player2_kr2', 'team2_player2_kr3', 'team2_player2_kr4', 'team2_player2_kr5', 'team2_player2_tok', 'team2_player2_tod', 'team2_player2_okr', 'team2_player2_okra', 'team2_player2_twp', 'team2_player2_fkwon', 'team2_player2_riflek', 'team2_player2_sniperk', 'team2_player2_smgk', 'team2_player2_pistolk', 'team2_player2_grenadek', 'team2_player2_otherk', 'team2_player3_rating', 'team2_player3_dpr', 'team2_player3_kast', 'team2_player3_impact', 'team2_player3_adr', 'team2_player3_kpr', 'team2_player3_kills', 'team2_player3_hs', 'team2_player3_deaths', 'team2_player3_apr', 'team2_player3_kd', 'team2_player3_dr', 'team2_player3_sbtr', 'team2_player3_stpr', 'team2_player3_gdmgr', 'team2_player3_maps', 'team2_player3_top5', 'team2_player3_top10', 'team2_player3_top20', 'team2_player3_top30', 'team2_player3_top50', 'team2_player3_kpd', 'team2_player3_rwk', 'team2_player3_kdd', 'team2_player3_kr0', 'team2_player3_kr1', 'team2_player3_kr2', 'team2_player3_kr3', 'team2_player3_kr4', 'team2_player3_kr5', 'team2_player3_tok', 'team2_player3_tod', 'team2_player3_okr', 'team2_player3_okra', 'team2_player3_twp', 'team2_player3_fkwon', 'team2_player3_riflek', 'team2_player3_sniperk', 'team2_player3_smgk', 'team2_player3_pistolk', 'team2_player3_grenadek', 'team2_player3_otherk', 'team2_player4_rating', 'team2_player4_dpr', 'team2_player4_kast', 'team2_player4_impact', 'team2_player4_adr', 'team2_player4_kpr', 'team2_player4_kills', 'team2_player4_hs', 'team2_player4_deaths', 'team2_player4_apr', 'team2_player4_kd', 'team2_player4_dr', 'team2_player4_sbtr', 'team2_player4_stpr', 'team2_player4_gdmgr', 'team2_player4_maps', 'team2_player4_top5', 'team2_player4_top10', 'team2_player4_top20', 'team2_player4_top30', 'team2_player4_top50', 'team2_player4_kpd', 'team2_player4_rwk', 'team2_player4_kdd', 'team2_player4_kr0', 'team2_player4_kr1', 'team2_player4_kr2', 'team2_player4_kr3', 'team2_player4_kr4', 'team2_player4_kr5', 'team2_player4_tok', 'team2_player4_tod', 'team2_player4_okr', 'team2_player4_okra', 'team2_player4_twp', 'team2_player4_fkwon', 'team2_player4_riflek', 'team2_player4_sniperk', 'team2_player4_smgk', 'team2_player4_pistolk', 'team2_player4_grenadek', 'team2_player4_otherk', 'team2_player5_rating', 'team2_player5_dpr', 'team2_player5_kast', 'team2_player5_impact', 'team2_player5_adr', 'team2_player5_kpr', 'team2_player5_kills', 'team2_player5_hs', 'team2_player5_deaths', 'team2_player5_apr', 'team2_player5_kd', 'team2_player5_dr', 'team2_player5_sbtr', 'team2_player5_stpr', 'team2_player5_gdmgr', 'team2_player5_maps', 'team2_player5_top5', 'team2_player5_top10', 'team2_player5_top20', 'team2_player5_top30', 'team2_player5_top50', 'team2_player5_kpd', 'team2_player5_rwk', 'team2_player5_kdd', 'team2_player5_kr0', 'team2_player5_kr1', 'team2_player5_kr2', 'team2_player5_kr3', 'team2_player5_kr4', 'team2_player5_kr5', 'team2_player5_tok', 'team2_player5_tod', 'team2_player5_okr', 'team2_player5_okra', 'team2_player5_twp', 'team2_player5_fkwon', 'team2_player5_riflek', 'team2_player5_sniperk', 'team2_player5_smgk', 'team2_player5_pistolk', 'team2_player5_grenadek', 'team2_player5_otherk', 'team1_player1_3m_rating', 'team1_player1_3m_dpr', 'team1_player1_3m_kast', 'team1_player1_3m_impact', 'team1_player1_3m_adr', 'team1_player1_3m_kpr', 'team1_player1_3m_kills', 'team1_player1_3m_hs', 'team1_player1_3m_deaths', 'team1_player1_3m_apr', 'team1_player1_3m_kd', 'team1_player1_3m_dr', 'team1_player1_3m_sbtr', 'team1_player1_3m_stpr', 'team1_player1_3m_gdmgr', 'team1_player1_3m_maps', 'team1_player1_3m_top5', 'team1_player1_3m_top10', 'team1_player1_3m_top20', 'team1_player1_3m_top30', 'team1_player1_3m_top50', 'team1_player1_3m_kpd', 'team1_player1_3m_rwk', 'team1_player1_3m_kdd', 'team1_player1_3m_kr0', 'team1_player1_3m_kr1', 'team1_player1_3m_kr2', 'team1_player1_3m_kr3', 'team1_player1_3m_kr4', 'team1_player1_3m_kr5', 'team1_player1_3m_tok', 'team1_player1_3m_tod', 'team1_player1_3m_okr', 'team1_player1_3m_okra', 'team1_player1_3m_twp', 'team1_player1_3m_fkwon', 'team1_player1_3m_riflek', 'team1_player1_3m_sniperk', 'team1_player1_3m_smgk', 'team1_player1_3m_pistolk', 'team1_player1_3m_grenadek', 'team1_player1_3m_otherk', 'team1_player2_3m_rating', 'team1_player2_3m_dpr', 'team1_player2_3m_kast', 'team1_player2_3m_impact', 'team1_player2_3m_adr', 'team1_player2_3m_kpr', 'team1_player2_3m_kills', 'team1_player2_3m_hs', 'team1_player2_3m_deaths', 'team1_player2_3m_apr', 'team1_player2_3m_kd', 'team1_player2_3m_dr', 'team1_player2_3m_sbtr', 'team1_player2_3m_stpr', 'team1_player2_3m_gdmgr', 'team1_player2_3m_maps', 'team1_player2_3m_top5', 'team1_player2_3m_top10', 'team1_player2_3m_top20', 'team1_player2_3m_top30', 'team1_player2_3m_top50', 'team1_player2_3m_kpd', 'team1_player2_3m_rwk', 'team1_player2_3m_kdd', 'team1_player2_3m_kr0', 'team1_player2_3m_kr1', 'team1_player2_3m_kr2', 'team1_player2_3m_kr3', 'team1_player2_3m_kr4', 'team1_player2_3m_kr5', 'team1_player2_3m_tok', 'team1_player2_3m_tod', 'team1_player2_3m_okr', 'team1_player2_3m_okra', 'team1_player2_3m_twp', 'team1_player2_3m_fkwon', 'team1_player2_3m_riflek', 'team1_player2_3m_sniperk', 'team1_player2_3m_smgk', 'team1_player2_3m_pistolk', 'team1_player2_3m_grenadek', 'team1_player2_3m_otherk', 'team1_player3_3m_rating', 'team1_player3_3m_dpr', 'team1_player3_3m_kast', 'team1_player3_3m_impact', 'team1_player3_3m_adr', 'team1_player3_3m_kpr', 'team1_player3_3m_kills', 'team1_player3_3m_hs', 'team1_player3_3m_deaths', 'team1_player3_3m_apr', 'team1_player3_3m_kd', 'team1_player3_3m_dr', 'team1_player3_3m_sbtr', 'team1_player3_3m_stpr', 'team1_player3_3m_gdmgr', 'team1_player3_3m_maps', 'team1_player3_3m_top5', 'team1_player3_3m_top10', 'team1_player3_3m_top20', 'team1_player3_3m_top30', 'team1_player3_3m_top50', 'team1_player3_3m_kpd', 'team1_player3_3m_rwk', 'team1_player3_3m_kdd', 'team1_player3_3m_kr0', 'team1_player3_3m_kr1', 'team1_player3_3m_kr2', 'team1_player3_3m_kr3', 'team1_player3_3m_kr4', 'team1_player3_3m_kr5', 'team1_player3_3m_tok', 'team1_player3_3m_tod', 'team1_player3_3m_okr', 'team1_player3_3m_okra', 'team1_player3_3m_twp', 'team1_player3_3m_fkwon', 'team1_player3_3m_riflek', 'team1_player3_3m_sniperk', 'team1_player3_3m_smgk', 'team1_player3_3m_pistolk', 'team1_player3_3m_grenadek', 'team1_player3_3m_otherk', 'team1_player4_3m_rating', 'team1_player4_3m_dpr', 'team1_player4_3m_kast', 'team1_player4_3m_impact', 'team1_player4_3m_adr', 'team1_player4_3m_kpr', 'team1_player4_3m_kills', 'team1_player4_3m_hs', 'team1_player4_3m_deaths', 'team1_player4_3m_apr', 'team1_player4_3m_kd', 'team1_player4_3m_dr', 'team1_player4_3m_sbtr', 'team1_player4_3m_stpr', 'team1_player4_3m_gdmgr', 'team1_player4_3m_maps', 'team1_player4_3m_top5', 'team1_player4_3m_top10', 'team1_player4_3m_top20', 'team1_player4_3m_top30', 'team1_player4_3m_top50', 'team1_player4_3m_kpd', 'team1_player4_3m_rwk', 'team1_player4_3m_kdd', 'team1_player4_3m_kr0', 'team1_player4_3m_kr1', 'team1_player4_3m_kr2', 'team1_player4_3m_kr3', 'team1_player4_3m_kr4', 'team1_player4_3m_kr5', 'team1_player4_3m_tok', 'team1_player4_3m_tod', 'team1_player4_3m_okr', 'team1_player4_3m_okra', 'team1_player4_3m_twp', 'team1_player4_3m_fkwon', 'team1_player4_3m_riflek', 'team1_player4_3m_sniperk', 'team1_player4_3m_smgk', 'team1_player4_3m_pistolk', 'team1_player4_3m_grenadek', 'team1_player4_3m_otherk', 'team1_player5_3m_rating', 'team1_player5_3m_dpr', 'team1_player5_3m_kast', 'team1_player5_3m_impact', 'team1_player5_3m_adr', 'team1_player5_3m_kpr', 'team1_player5_3m_kills', 'team1_player5_3m_hs', 'team1_player5_3m_deaths', 'team1_player5_3m_apr', 'team1_player5_3m_kd', 'team1_player5_3m_dr', 'team1_player5_3m_sbtr', 'team1_player5_3m_stpr', 'team1_player5_3m_gdmgr', 'team1_player5_3m_maps', 'team1_player5_3m_top5', 'team1_player5_3m_top10', 'team1_player5_3m_top20', 'team1_player5_3m_top30', 'team1_player5_3m_top50', 'team1_player5_3m_kpd', 'team1_player5_3m_rwk', 'team1_player5_3m_kdd', 'team1_player5_3m_kr0', 'team1_player5_3m_kr1', 'team1_player5_3m_kr2', 'team1_player5_3m_kr3', 'team1_player5_3m_kr4', 'team1_player5_3m_kr5', 'team1_player5_3m_tok', 'team1_player5_3m_tod', 'team1_player5_3m_okr', 'team1_player5_3m_okra', 'team1_player5_3m_twp', 'team1_player5_3m_fkwon', 'team1_player5_3m_riflek', 'team1_player5_3m_sniperk', 'team1_player5_3m_smgk', 'team1_player5_3m_pistolk', 'team1_player5_3m_grenadek', 'team1_player5_3m_otherk', 'team2_player1_3m_rating', 'team2_player1_3m_dpr', 'team2_player1_3m_kast', 'team2_player1_3m_impact', 'team2_player1_3m_adr', 'team2_player1_3m_kpr', 'team2_player1_3m_kills', 'team2_player1_3m_hs', 'team2_player1_3m_deaths', 'team2_player1_3m_apr', 'team2_player1_3m_kd', 'team2_player1_3m_dr', 'team2_player1_3m_sbtr', 'team2_player1_3m_stpr', 'team2_player1_3m_gdmgr', 'team2_player1_3m_maps', 'team2_player1_3m_top5', 'team2_player1_3m_top10', 'team2_player1_3m_top20', 'team2_player1_3m_top30', 'team2_player1_3m_top50', 'team2_player1_3m_kpd', 'team2_player1_3m_rwk', 'team2_player1_3m_kdd', 'team2_player1_3m_kr0', 'team2_player1_3m_kr1', 'team2_player1_3m_kr2', 'team2_player1_3m_kr3', 'team2_player1_3m_kr4', 'team2_player1_3m_kr5', 'team2_player1_3m_tok', 'team2_player1_3m_tod', 'team2_player1_3m_okr', 'team2_player1_3m_okra', 'team2_player1_3m_twp', 'team2_player1_3m_fkwon', 'team2_player1_3m_riflek', 'team2_player1_3m_sniperk', 'team2_player1_3m_smgk', 'team2_player1_3m_pistolk', 'team2_player1_3m_grenadek', 'team2_player1_3m_otherk', 'team2_player2_3m_rating', 'team2_player2_3m_dpr', 'team2_player2_3m_kast', 'team2_player2_3m_impact', 'team2_player2_3m_adr', 'team2_player2_3m_kpr', 'team2_player2_3m_kills', 'team2_player2_3m_hs', 'team2_player2_3m_deaths', 'team2_player2_3m_apr', 'team2_player2_3m_kd', 'team2_player2_3m_dr', 'team2_player2_3m_sbtr', 'team2_player2_3m_stpr', 'team2_player2_3m_gdmgr', 'team2_player2_3m_maps', 'team2_player2_3m_top5', 'team2_player2_3m_top10', 'team2_player2_3m_top20', 'team2_player2_3m_top30', 'team2_player2_3m_top50', 'team2_player2_3m_kpd', 'team2_player2_3m_rwk', 'team2_player2_3m_kdd', 'team2_player2_3m_kr0', 'team2_player2_3m_kr1', 'team2_player2_3m_kr2', 'team2_player2_3m_kr3', 'team2_player2_3m_kr4', 'team2_player2_3m_kr5', 'team2_player2_3m_tok', 'team2_player2_3m_tod', 'team2_player2_3m_okr', 'team2_player2_3m_okra', 'team2_player2_3m_twp', 'team2_player2_3m_fkwon', 'team2_player2_3m_riflek', 'team2_player2_3m_sniperk', 'team2_player2_3m_smgk', 'team2_player2_3m_pistolk', 'team2_player2_3m_grenadek', 'team2_player2_3m_otherk', 'team2_player3_3m_rating', 'team2_player3_3m_dpr', 'team2_player3_3m_kast', 'team2_player3_3m_impact', 'team2_player3_3m_adr', 'team2_player3_3m_kpr', 'team2_player3_3m_kills', 'team2_player3_3m_hs', 'team2_player3_3m_deaths', 'team2_player3_3m_apr', 'team2_player3_3m_kd', 'team2_player3_3m_dr', 'team2_player3_3m_sbtr', 'team2_player3_3m_stpr', 'team2_player3_3m_gdmgr', 'team2_player3_3m_maps', 'team2_player3_3m_top5', 'team2_player3_3m_top10', 'team2_player3_3m_top20', 'team2_player3_3m_top30', 'team2_player3_3m_top50', 'team2_player3_3m_kpd', 'team2_player3_3m_rwk', 'team2_player3_3m_kdd', 'team2_player3_3m_kr0', 'team2_player3_3m_kr1', 'team2_player3_3m_kr2', 'team2_player3_3m_kr3', 'team2_player3_3m_kr4', 'team2_player3_3m_kr5', 'team2_player3_3m_tok', 'team2_player3_3m_tod', 'team2_player3_3m_okr', 'team2_player3_3m_okra', 'team2_player3_3m_twp', 'team2_player3_3m_fkwon', 'team2_player3_3m_riflek', 'team2_player3_3m_sniperk', 'team2_player3_3m_smgk', 'team2_player3_3m_pistolk', 'team2_player3_3m_grenadek', 'team2_player3_3m_otherk', 'team2_player4_3m_rating', 'team2_player4_3m_dpr', 'team2_player4_3m_kast', 'team2_player4_3m_impact', 'team2_player4_3m_adr', 'team2_player4_3m_kpr', 'team2_player4_3m_kills', 'team2_player4_3m_hs', 'team2_player4_3m_deaths', 'team2_player4_3m_apr', 'team2_player4_3m_kd', 'team2_player4_3m_dr', 'team2_player4_3m_sbtr', 'team2_player4_3m_stpr', 'team2_player4_3m_gdmgr', 'team2_player4_3m_maps', 'team2_player4_3m_top5', 'team2_player4_3m_top10', 'team2_player4_3m_top20', 'team2_player4_3m_top30', 'team2_player4_3m_top50', 'team2_player4_3m_kpd', 'team2_player4_3m_rwk', 'team2_player4_3m_kdd', 'team2_player4_3m_kr0', 'team2_player4_3m_kr1', 'team2_player4_3m_kr2', 'team2_player4_3m_kr3', 'team2_player4_3m_kr4', 'team2_player4_3m_kr5', 'team2_player4_3m_tok', 'team2_player4_3m_tod', 'team2_player4_3m_okr', 'team2_player4_3m_okra', 'team2_player4_3m_twp', 'team2_player4_3m_fkwon', 'team2_player4_3m_riflek', 'team2_player4_3m_sniperk', 'team2_player4_3m_smgk', 'team2_player4_3m_pistolk', 'team2_player4_3m_grenadek', 'team2_player4_3m_otherk', 'team2_player5_3m_rating', 'team2_player5_3m_dpr', 'team2_player5_3m_kast', 'team2_player5_3m_impact', 'team2_player5_3m_adr', 'team2_player5_3m_kpr', 'team2_player5_3m_kills', 'team2_player5_3m_hs', 'team2_player5_3m_deaths', 'team2_player5_3m_apr', 'team2_player5_3m_kd', 'team2_player5_3m_dr', 'team2_player5_3m_sbtr', 'team2_player5_3m_stpr', 'team2_player5_3m_gdmgr', 'team2_player5_3m_maps', 'team2_player5_3m_top5', 'team2_player5_3m_top10', 'team2_player5_3m_top20', 'team2_player5_3m_top30', 'team2_player5_3m_top50', 'team2_player5_3m_kpd', 'team2_player5_3m_rwk', 'team2_player5_3m_kdd', 'team2_player5_3m_kr0', 'team2_player5_3m_kr1', 'team2_player5_3m_kr2', 'team2_player5_3m_kr3', 'team2_player5_3m_kr4', 'team2_player5_3m_kr5', 'team2_player5_3m_tok', 'team2_player5_3m_tod', 'team2_player5_3m_okr', 'team2_player5_3m_okra', 'team2_player5_3m_twp', 'team2_player5_3m_fkwon', 'team2_player5_3m_riflek', 'team2_player5_3m_sniperk', 'team2_player5_3m_smgk', 'team2_player5_3m_pistolk', 'team2_player5_3m_grenadek', 'team2_player5_3m_otherk']
        
        csv_content = []
        
        upcomingMatches = parseUpcomingMatches(-1) # 3d array
        
        for match in upcomingMatches:
            csv_row = [match[0], match[1], match[2], match[3], match[4]] # date, event, url, team1, team2
            player_stats = []
            player_stats_3m = []

            for player in match[5]: # for each player
                player_stats = parsePlayerProfile(player, "2017-01-01", datetime.now().strftime('%Y-%m-%d'))
                player_stats_3m = parsePlayerProfile(player, (datetime.now() + relativedelta(months=-3)).strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'))
                
                if player_stats is None or player_stats_3m is None:
                    if DEBUG == 1:
                        print("\nSkipping match because " + player + " is lacking stats.")
                    break
                             
                csv_row.extend(player_stats)
                csv_row.extend(player_stats_3m)
            
            if len(csv_row) == 845: # add match row only if we have all data
                csv_content.append(csv_row)

        with open(sys.path[0] + '/upcoming_matches.csv', 'a', newline='') as f:
            df = pd.DataFrame(csv_content, columns=csv_headers)
            df.to_csv(f, mode='a', index=False, header=not f.tell()) # write or append
        
        if DEBUG == 1:
            print()
            print(df)

    if task == 2:
        print("Coming soon to your city.")