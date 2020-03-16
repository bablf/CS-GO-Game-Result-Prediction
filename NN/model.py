import csv
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, from_numpy, unsqueeze


from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import scale

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
        reader = csv.reader(scraped, delimiter=';')
        first_row = next(reader)  # skip to second line, because first doent have values
        # for debugging
        # print(np.array(first_row[5:-1]).reshape(2, 5, 4))
        ys = []
        data_x,data_y = [],[]
        for row in reader:
            if row:  # avoid blank lines
                y = float(row[-1]) # take Goldlabel
                ys.append(y)
                winner = [0.0, 1.0] if y == 2.0 else [1.0, 0.0]
                x = [0.0 if elem == "" else float(elem) for elem in row[5:-1]]
                NUMB_ROWS += 1
                data_x.append(np.array(x))
                data_y.append(np.array(winner))
                if first:
                    first = False
                    NUMB_FEAT = len(x)

        print(NUMB_FEAT)
        print(sum([1 for y in ys if y == 2.0])/NUMB_ROWS)

        data_x = scale(np.array(data_x)).reshape(NUMB_ROWS, 2, 5, int(NUMB_FEAT/10))
        return data_x, data_y



def import_csv(csvfilename): # https://stackoverflow.com/a/53483446
    """
    This funciton takes a csv file and returns a dataset looking like this
    [(match_matrix, winner),  ... ]
    """
    data_x, data_y = [],[] # Feature
    with open(csvfilename, "r", encoding="utf-8", errors="ignore") as scraped:
        reader = csv.reader(scraped, delimiter=';')
        first_row = next(reader)  # skip to second line, because first doent have values

        for row in reader:
            print(row)
            team1, team2 = [],[]
            if row:  # avoid blank lines
                y = float(row[-1]) # take Goldlabel

                winner = [0.0, 1.0] if y == 2.0 else [1.0, 0.0]
                x = [0.0 if elem == "" else float(elem) for elem in row[5:-1]] # remove "-" und set to float
                t1 = x[:20]
                t2 = x[20:]
                for i in range(0,5):    # group all player features
                    player = t1[i::5]
                    team1.append(player)
                for i in range(0,5):    # group all player features
                    player = t2[i::5]
                    team2.append(player)

            data_x.append(np.array([team1,team2]))
            data_y.append(np.array(winner))

        return np.array(data_x), np.array(data_y)

class DatasetIterator():
    """
    berechnet den Goldlabel-Satzvektor "on demand"
    """
    def __init__(self, dataset, batchsize):
        self.matches, self.winners = dataset
        self.batchsize = batchsize

    def __iter__(self):
        # !!!!!!!!!!!!!!!!  DO NOT DELETE COMMENTS !!!!!!!!!!!!!!!!!!
        #matches, winners = [], []
        #matches = [item[0] for item in self.dataset]
        #winner = [item[1] for item in self.dataset]
        #matches = np.array(matches)
        #winner = np.array(winner)

        #for i in range(0, len(matches), batchsize):
        #    yield from_numpy(matches[i:i + batchsize]).float(), from_numpy(winner[i:i + batchsize]).float()

        for match, winner in zip(self.matches,self.winners):
            yield from_numpy(match).float(), from_numpy(winner).float()

    def __len__(self):
        return len(self.matches)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        """
        in_channels (python:int) – Number of channels in the input image
        out_channels (python:int) – Number of channels produced by the convolution
        kernel_size (python:int or tuple) – Size of the convolving kernel
        """
        self.Convolution = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=(5,1)) # Filter muss 1xNUMB_PLAYER sein
        self.Dense64 = nn.Linear(4, 64) # (2x NUMB_FEAT)
        self.Dense16 = nn.Linear(64,16)
        self.Dense1  = nn.Linear(16,1)
        self.Sigmoid = nn.Sigmoid()
        self.Dropout = nn.Dropout(p=0.2)


    def forward(self, matches):
        team1, team2 = matches
        team1 = unsqueeze(unsqueeze(team1,0),0)
        team2 = unsqueeze(unsqueeze(team2,0),0)
        #conv_out = self.Convolution(matches)
        conv_out1 = self.Convolution(team1).view(1,4) # TODO: NUMB_FEAT statt 4
        conv_out2 = self.Convolution(team2).view(1,4) # TODO: NUMB_FEAT statt 4
        # concat the matrices:
        team_feature = torch.cat((conv_out1, conv_out2),0)
        drop_feat = self.Dropout(team_feature)

        x64 = torch.tanh(self.Dense64(drop_feat))
        drop = self.Dropout(x64)

        x16 = torch.tanh(self.Dense16(drop))
        drop = self.Dropout(x16)

        #x = torch.tanh(self.Dense1(x16)).view(2)
        #prediction = self.Sigmoid(self.Dense1(x16).view(2))
        prediction = torch.tanh(self.Dense1(x16).view(2))
        #print(prediction)
        #prediction = self.Sigmoid(x)
        #prediction = nn.Sigmoid()(self.Dense1(x16)) #only one of the two
        #return torch.tensor(max(prediction))
        return prediction # (P(Home-Won), P(Away-Won))

    def save_model(self, model, model_filepath='model.pkl'):
        """
        Speichert das model + parameter
        """
        print("===== Model wird gespeichert ======\n")
        torch.save(model.state_dict(), model_filepath)

if __name__ == "__main__" :
# Values can be changed, to (maybe) improve perfermance a bit
    learning_rate = 0.001
    epochs = 10
    batchsize = 200 # not implemented

    model_filepath = "./model.pkl"
    scale_data("../scraperinusTotalicus/past_matches.csv")
    exit()
    print("===== Daten werden gelesen======\n")
    data_x, data_y = import_csv("../scraperinusTotalicus/past_matches.csv")
    print("===== Daten eingelesen ======")
    print(data_x)
    # Start training
    ezBetticus = Model()
    train(ezBetticus, data_x, data_y, epochs, batchsize, learning_rate,  model_filepath) # train and (will) save the best model EUWEST