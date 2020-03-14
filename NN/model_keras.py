import csv
import random
import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def import_csv(csvfilename): # https://stackoverflow.com/a/53483446
    """
    This funciton takes a csv file and returns a dataset
    """
    data_x, data_y = [],[]
    row_index = 1
    with open(csvfilename, "r", encoding="utf-8", errors="ignore") as scraped:
        reader = csv.reader(scraped, delimiter=';')
        first_row = next(reader)  # skip to second line, because first doent have values
        for row in reader:
            team1, team2 = [],[]
            if row:  # avoid blank lines
                y = float(row[-1]) # take Goldlabel
                winner = 1.0 if y == 2.0 else 0.0
                x = [0.0 if elem == "" else float(elem) for elem in row[5:-1]] # remove "-" und set to float
                t1 = x[:20]
                t2 = x[20:]
                for i in range(0,5):    # group all player features
                    player = t1[i::5]
                    team1.append(player)
                for i in range(0,5):    # group all player features
                    player = t2[i::5]
                    team2.append(player)

            data_x.append((np.array([team1,team2])))
            data_y.append(winner)

        return np.array(data_x), np.array(data_y)

def save_model(model, model_filepath='model.pkl', data_filepath='data_params.pkl'):
    """
    Speichert das model + parameter
    """
    torch.save(model.state_dict(), model_filepath)


def create_Model():
    model = Sequential()
    model.add(Conv2D(4, kernel_size=(1, 5), strides=1,
                     input_shape= (2, 5, 4)))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(1, activation='tanh'))
    return model

if __name__ == "__main__" :
# Values can be changed, to (maybe) improve perfermance a bit
    learning_rate = 0.0001
    epochs = 20
    batchsize = 200

    model_file = "./data/model.pkl"
    print("===== Daten werden gelesen======\n")
    data_x, data_y = import_csv("./data/past_matches.csv")
    print("===== Daten eingelesen ======")
    # split into train and test data
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.33, random_state=1337)
    print(x_train.shape)
    ezBetticus = create_Model()
    ezBetticus.compile(loss='mean_squared_error',
                        optimizer=optimizers.SGD(lr=0.001),
                        metrics=['accuracy'])
    ezBetticus.fit(x_train, y_train,
          batch_size=batchsize,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
