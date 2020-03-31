import csv
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, from_numpy, unsqueeze

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

from model import Model, DatasetIterator

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
        teams = []
        reader = csv.reader(scraped, delimiter=',')
        first_row = next(reader)  # skip to second line, because first doent have values
        # for debugging
        # print(np.array(first_row[5:-1]).reshape(2, 5, 4))

        data_x,data_y = [],[]
        for row in reader:
            if row:  # avoid blank lines
                teams.append(row[3])
                teams.append(row[4])
                y = float(row[-1]) # take Goldlabel
                winner = [0.0, 1.0] if y == 2.0 else [1.0, 0.0]
                x = [0.0 if elem == "" else float(elem) for elem in row[5:-1]]
                NUMB_ROWS += 1
                data_x.append(np.array(x))
                data_y.append(np.array(winner))
                if first:
                    first = False
                    NUMB_FEAT = len(x)

        data_x = scale(np.array(data_x)).reshape(NUMB_ROWS, 2, 5, int(NUMB_FEAT/10))

        return data_x, data_y, teams


def converter(preds):
    if preds[0] > preds[1]: # home team won
        return torch.tensor([1.0, 0.0])
    elif preds[0] < preds[1]: # away team won
        return torch.tensor([0.0, 1.0])
    else: return torch.tensor([2.0,2.0]) # same size no correct clasification

def calc_acc(preds, goldlabel):
    right = 0
    for pred, gold in zip(preds, goldlabel):
        if torch.equal(pred, gold):
            right +=1
    return right/len(preds)

def train(model, data_x, data_y, epochs, batchsize, learning_rate, model_filepath):

    opt = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.001) # Optimizer = Stocastic Gradient Descent
    loss_func = nn.MSELoss(reduction='sum') # loss-Function = Sum Squard Error
    old_accuracy = 0.0
    for epoch in range(1, epochs + 1): # Epochen die iteriert werden soll
        i, running_loss = 0, 0
        #list_train_preds, list_train_goldlabel = [],[]
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.33) # Shuffle Matches
        train_iter = DatasetIterator((x_train, y_train), batchsize) # für alle Matches (yield)
        model.train()  # turn on training mode
        for match, goldlabel in tqdm(train_iter):
            opt.zero_grad()
            predictions = model(match)
            #list_train_preds.append(predictions)
            #list_train_goldlabel.append(goldlabel)
            loss = loss_func(predictions, goldlabel)
            loss.backward()
            opt.step()
        #train_accuracy = calc_acc(list_train_preds, list_train_goldlabel)
        #print("Train accuracy = ", train_accuracy)


        model.eval()  # turn on evaluation mode
        print("\n--- EVALUIERUNG ---")
        test_iter = DatasetIterator((x_test,y_test),batchsize)
        list_preds, list_goldlabel = [],[]
        for match, goldlabel in tqdm(test_iter):
            # matches = matches.to(device)
            # goldlabels = goldlabels.to(device)

            predictions = model(match)
            loss = loss_func(predictions, goldlabel)
            running_loss += loss.item()

            list_preds.append(converter(predictions))
            list_goldlabel.append(goldlabel)
            i += 1
            if i % 2000 == 1999:    # print every 2000 matches
                accuracy = calc_acc(list_preds, list_goldlabel)
                list_preds, list_goldlabel = [],[]
                print('[%d, %5d] loss: %.3f acc: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000, accuracy))
                running_loss = 0.0
                if accuracy > old_accuracy:
                    model.save_model(model, model_filepath)
                    old_accuracy = accuracy


        print("\nEpoche fertig {}/{},".format(epoch, epochs))


if __name__ == "__main__" :
# Values can be changed, to (maybe) improve perfermance a bit
    learning_rate = 0.001
    epochs = 10
    batchsize = 200 # not implemented

    model_filepath = "./model.pkl"
    #scale_data("../scraperinusTotalicus/past_matches.csv")

    print("===== Daten werden gelesen======\n")
    data_x, data_y, teams = scale_data("../scraperinusTotalus/past_matches_2017-04-01_2020-03-30.csv")
    print("===== Daten eingelesen =========")
    print(data_x.shape)
    # Start training
    ezBetticus = Model()
    train(ezBetticus, data_x, data_y, epochs, batchsize, learning_rate,  model_filepath) # train and (will) save the best model EUWEST
