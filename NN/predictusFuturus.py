

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
                x = [0.0 if elem == "" else float(elem) for elem in row[5:-1]]
                NUMB_ROWS += 1
                data_x.append(np.array(x))
                data_y.append(np.array(winner))
                if first:
                    first = False
                    NUMB_FEAT = len(x)

        data_x = scale(np.array(data_x)).reshape(NUMB_ROWS, 2, 5, int(NUMB_FEAT/10))

        return data_x, data_y, teams


def predictusFuturus():
    for match, goldlabel in tqdm(train_iter):
        opt.zero_grad()
        predictions = model(match)


if __name__ == "__main__" :
    #TODO:  load model
    # load upcoming matches
    # call predictusFuturus()
    # DatasetIterator umschreiben
