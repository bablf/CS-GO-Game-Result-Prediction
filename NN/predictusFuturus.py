import csv
import torch
import numpy as np

from torch import from_numpy
from tqdm import tqdm
from model import Model
from prettytable import PrettyTable
from sklearn.preprocessing import scale
from portfolioMaxicus import calc_portfolio


class DatasetIterator():
    """
    berechnet den Goldlabel-Satzvektor "on demand"
    """
    def __init__(self, dataset):
        self.matches = dataset

    def __iter__(self):
        for match in self.matches:
            yield from_numpy(match).float()

    def __len__(self):
        return len(self.matches)

def list2tuples(L):
    # Turns list into list of tuples
    it = iter(L)
    L = zip(it, it)
    return L

def scale_data(csvfilename):
    """
    This method reads the data and scales it.

    Important:
    Only works if features are sorted playerwise. Way faster than import_csv.

    This method should work on the new features.
    """
    with open(csvfilename, "r", encoding="utf-8", errors="ignore") as scraped:
        NUMB_ROWS, NUMB_FEAT = 0, 0
        first = True
        teams, odds = [], []
        reader = csv.reader(scraped, delimiter=',')
        first_row = next(reader)  # skip to second line, because first doent have values
        # for debugging

        data_x = []
        for row in reader:
            if row:  # avoid blank lines
                teams.append(row[3])
                teams.append(row[4])

                x = [0.0 if elem == "" else float(elem) for elem in row[5:-2]]
                NUMB_ROWS += 1
                data_x.append(np.array(x))
                odds.append(float(row[-2]))
                odds.append(float(row[-1]))
                if first:
                    first = False
                    NUMB_FEAT = len(x)

        data_x = scale(np.array(data_x)).reshape(NUMB_ROWS, 2, 5, int(NUMB_FEAT/10))

        return data_x, teams, odds


def predictus_futurus(data_x):
    predictions = []
    train_iter = DatasetIterator(data_x)
    for match in tqdm(train_iter):
        pred = ezBetticus(match)
        predictions.extend(pred.tolist())

    return predictions

def calc_min_turnover(bets, odds):
    betReturns = []
    minTurnover = 0.0
    for b, o in zip(list2tuples(bets), list2tuples(odds)):
        if b[0] > 0.0 and b[1] > 0.0: # wenn auf beide Ausgänge gewettet wurde
            minTurnover += min([b[0]*o[0], b[1]*o[1]]) # den mit dem geringeren Ertrag hinzufügen

    return minTurnover

if __name__ == "__main__" :
    """
    TODO:
    1. diese main aufräumen. 3 Funktionen schreiben
    2. nach args fragen: max bet size pro spiel, wager insgesamt
    3. in model file einlesen für alle Betriebsysteme verfügbar machen.
    4. Mit tol morgen weiter machen.
    5. threshold auf 0.2 festlegen. Ergibt sonst keinen Sinn.
    6. ezBetticus verbessern.
    """

    #ezBetticus = load('model.pkl')
    ezBetticus = Model()
    ezBetticus.load_state_dict(torch.load("model.pkl"))
    ezBetticus.eval()

    # load upcoming matches, predict Winners and make Portfolio
    data_x, teams, odds = scale_data("../scraperinusTotalus/upcoming_matches_28-03-2020_11-53-47.csv")
    print("\n==== Sage Gewinner voraus ====")
    predictions = predictus_futurus(data_x)
    print("=== Berechne Wettstrategie ===")
    bets, expProfit, preds, odds, teams = calc_portfolio(predictions, odds,teams)

    if len(teams) % 2 != 0:
        print("Team missing.")
        exit()

    #build Matchup Table
    matchups = PrettyTable()
    matchups.field_names = ["Matchup", "Bet on T1", "Bet on T2", "Odds T1", "Odds T2", "WK-NN T1", "WK-NN T2"]
    for t, o, p, b in zip(list2tuples(teams), list2tuples(odds), list2tuples(preds), list2tuples(bets)):
        matchups.add_row([" vs. ".join(t), b[0], b[1], o[0], o[1], '{:.2%}'.format(p[0]), '{:.2%}'.format(p[1])])

    # build Finance Table
    minTurnover = calc_min_turnover(bets, odds)
    finance = PrettyTable()
    finance.field_names = ["Wager Size", "expected profit", "min. Turnover", "max. Loss"]
    finance.add_row([sum(bets), expProfit, minTurnover, minTurnover - sum(bets)])

    # print Tables
    print("\n\n=== So sollte gewettet werden ===")
    print(matchups.get_string())
    print("Finanzielle Informationen")
    print(finance.get_string())
