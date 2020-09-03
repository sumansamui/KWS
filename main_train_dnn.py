import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

import config


def load_data(data_path):
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data["logmel"])
    y = np.array(data["labels"])

    print("Data succesfully loaded!")

    return  X, y


if __name__ == "__main__":

    # load data
    X, y = load_data(config.DATA_PATH)

    print(X.shape)

    print(y.shape)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    print(X_train.shape)

    print(y_train.shape)

    print(X_test.shape)

    print(y_test.shape) 

    # build network topology

    # <Add your code>
    
    # compile model
    
    # train model

    # Evaluate model 
   