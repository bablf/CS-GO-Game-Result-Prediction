import csv
import torch
import numpy as np

from torch import from_numpy
from tqdm import tqdm
from model import Model
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


def predictusFuturus(data_x):

    predictions = []
    train_iter = DatasetIterator(data_x)
    for match in tqdm(train_iter):
        pred = ezBetticus(match)
        predictions.extend(pred.tolist())

    return predictions


if __name__ == "__main__" :

    #ezBetticus = load('model.pkl')
    ezBetticus = Model()
    ezBetticus.load_state_dict(torch.load("model.pkl"))
    ezBetticus.eval()

    # load upcoming matches
    data_x, teams, odds = scale_data("../scraperinusTotalus/upcoming_matches_28-03-2020_11-53-47.csv")
    predictions = predictusFuturus(data_x)
    print(teams)
    print(predictions)
    print(odds)
    calc_portfolio(predictions, odds)
