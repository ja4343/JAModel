from math import floor

import pandas as pd
import numpy as np
import traceback
import pickle

class Model(object):
    """
    Abstract base model class defining the model interface - a model is an object with a training method and a testing/forecasting ability given some input data
    """
    def __init__(self):
        self.__meta__ = "from ja_model library 2018"
        self.__data__ = {'X_train' : pd.DataFrame({}), 'Y_train' : pd.Series([]), 'X_val' : pd.DataFrame({}), 'Y_val' : pd.Series([]), 'X_test' : pd.DataFrame({}), 'Y_test' : pd.Series([]), 'Y_forecast': pd.Series({})}
        self.save_count = 0


    def set_data(self, data_dictionary=None):
        """
        Allows the user to specify their own training, validation, and testing datasets.
        Args:
            data_dictionary : dictionary - dictionary of data objects containing keys in the list ['X_train', 'Y_train', 'X_val', 'Y_val', 'X_test', 'Y_test']
        """
        if data_dictionary == None:
            return 0
        else:
            try:
                for key in data_dictionary.keys():
                    if key not in ['X_train', 'Y_train', 'X_val', 'Y_val', 'X_test', 'Y_test']:
                        raise KeyError('ERROR: ' + key + ' is an invalid key, data has not been updated. Please choose from ["X_train", "Y_train", "X_val", "Y_val", "X_test", "Y_test"]')
                    self.__data__[key] = data_dictionary[key]
            except KeyError as ke:
                print(ke.args[0])


    def generate_split(self, X, Y, train_size=0.7, val_size=0.15, test_size=0.15):
        """
        Generates the random train/validation/test split for the input data X, Y without breaking the time-series order

        Default split is (train: 0.7, validation: 0.15, test: 0.15)

        Args:
            X : pd.DataFrame, pd.Series or list - input feature data
            Y : pd.DataFrame, pd.Series or list - input observables data
            train_size : float - proportion of data for training
            val_size : float - proportion of data for validation
            test_size : float - proportion of data for testing

        Note: train_size, val_size, and test_size should sum to 1.0
        """
        try:
            if (train_size + val_size + test_size != 1.0):
                raise ValueError("ERROR: train_size, val_size, and test_size should sum to 1.0, resetting classifier data.")
            try:
                length = len(Y)
            except TypeError:
                raise TypeError("ERROR: Input data not of the correct form: X (pd.DataFrame, pd.Series or list), Y (pd.Series or list)")
            train_idx = int(floor(train_size * length))
            val_idx = int(floor((train_size + val_size) * length))

            if isinstance(X, pd.DataFrame):
                X_train = X.iloc[:train_idx, :]
                X_val   = X.iloc[train_idx : val_idx, :]
                X_test  = X.iloc[val_idx : , :]
            elif isinstance(X, pd.Series):
                X_train = X.iloc[: train_idx]
                X_val   = X.iloc[train_idx : val_idx]
                X_test  = X.iloc[val_idx :]
            elif (isinstance(X, list) and not (isinstance(X[0], list)))  or (isinstance(X, np.ndarray) and (len(X.shape) == 1)):
                X_train = X[: train_idx]
                X_val   = X[train_idx : val_idx]
                X_test  = X[val_idx :]
            elif (isinstance(X, np.ndarray) and (len(X.shape) != 1)):
                X_train = X[:train_idx, :]
                X_val   = X[train_idx : val_idx, :]
                X_test  = X[val_idx : , :]
            elif (isinstance(X, list) and (len(X[0]) != 1)):
                X_train = X[:train_idx][:]
                X_val   = X[train_idx : val_idx][:]
                X_test  = X[val_idx :][:]
            else:
                raise TypeError("ERROR: Input data not of the correct form: X (pd.DataFrame, pd.Series or list), Y (pd.DataFrame, pd.Series or list)")

            if isinstance(Y, pd.DataFrame):
                Y_train = Y.iloc[:train_idx, :]
                Y_val   = Y.iloc[train_idx : val_idx, :]
                Y_test  = Y.iloc[val_idx : , :]
            elif isinstance(Y, pd.Series):
                Y_train = Y.iloc[: train_idx]
                Y_val   = Y.iloc[train_idx : val_idx]
                Y_test  = Y.iloc[val_idx :]
            elif isinstance(Y, list) or isinstance(Y, np.ndarray):
                Y_train = Y[: train_idx]
                Y_val   = Y[train_idx : val_idx]
                Y_test  = Y[val_idx :]
            else:
                raise TypeError("ERROR: Input data not of the correct form: X (pd.DataFrame, pd.Series or list), Y (pd.Series or list)")

            self.__data__['X_train'] = X_train
            self.__data__['Y_train'] = Y_train
            self.__data__['X_val'] = X_val
            self.__data__['Y_val'] = Y_val
            self.__data__['X_test'] = X_test
            self.__data__['Y_test'] = Y_test
            print('----------Finished generating training, validation, and testing data----------')

        except ValueError as ve:
            self.__data__ = {'X_train' : pd.DataFrame({}), 'Y_train' : pd.Series([]), 'X_val' : pd.DataFrame({}), 'Y_val' : pd.Series([]), 'X_test' : pd.DataFrame({}), 'Y_test' : pd.Series([])}
            print(ve.args[0])
        except TypeError as te:
            self.__data__ = {'X_train' : pd.DataFrame({}), 'Y_train' : pd.Series([]), 'X_val' : pd.DataFrame({}), 'Y_val' : pd.Series([]), 'X_test' : pd.DataFrame({}), 'Y_test' : pd.Series([])}
            print(te.args[0])

    def get_best_features(self):
        """
        Returns the most up date optimised feature set. If feature selection has not taken place, then the original feature set is used.

        Returns:
            features : list - best feature set on validation data
        """
        return self.best_features

    def get_data(self, data = None):
        """
        Getter function to extract the current training, validation, and test data sets for the classifier
        Args:
            data : str - choose which data to extract from ('X_train', 'Y_train', 'X_val', 'Y_val', 'X_test', 'Y_test')
        Returns:
            dictionary - dictionary of train, validation and test data for the given classifier if no data given

            or

            datatype - returns the data object referenced above

        """
        try:
            if data == None:
                return self.__data__
            elif data in ['X_train', 'X_val', 'X_test', 'Y_train', 'Y_val', 'Y_test']:
                return self.__data__[data]
            else:
                raise ValueError("Please choose one data type from ('X_train', 'Y_train', 'X_val', 'Y_val', 'X_test', 'Y_test')")

        except ValueError as ve:
            print(ve.args[0])

    def save(self, filename):
        """
        Saves intermediate model using pickle.
        Args:
            filename : str - target filename, usally .txt file
        """
        pickle.dump(self.__dict__, open(filename, "wb"))

    def load(self, filename):
        """
        Loads previously saved model using pickle.
        Args:
            filename : str - target filename, usally .txt file
        """
        try:
            self.__dict__ = pickle.load(open(filename, "rb"))
        except:
            print("ERROR: Error loading model from " + filename)
