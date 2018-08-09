import numpy as np
import seaborn as sns
from math import sqrt
from sklearn.metrics import mean_squared_error
import pandas as pd
from statsmodels.graphics.gofplots import qqplot
import pprint
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot

from ja_model.utils import _test_metric, _rmse, powerset, simpleaxis, x_label_setter, y_label_setter, nn_out_to_list
from ja_model.models import Model

import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import itertools
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout, GlobalAveragePooling1D
from keras.models import model_from_json

import pickle

class SLP(Model):
    def __init__(self, features, *args, **kwargs):
        """
        Args:
            features : list - list of desired features, these should match the headings of the training and testing data used later in the classification

        For more information, see LinReg.help()
        """
        super().__init__(*args, **kwargs)
        print("Initialising Single Layer Perceptron (Neural Network) Model")
        print("===========================================================")
        print(" ")
        print("Feature Set")
        print("-----------")
        print(" ")
        pprint.pprint(features)
        print(" ")
        print("IMPORTANT: For best results using the SLP class, input data should be normalised to lie within the same range e.g. [0,1]. Non-normalised data could lead to poor results.")
        self.input_features = features
        self.best_features = features
        self._current_features = features
        self.history = None
        self.__result = None
        self._model = None
        self.save_count = 0

    def get_latest_params(self):
        """
        Returns the latest optimised hyperparaneters

        Returns:
            parameters : dictionary - dictionary of hyperparameters
        """
        return self._latest_params

    def train(self, features=None, epochs=500, units=64, activation='relu', dropout = 0.0, batch_size = 16, save=True, load=False, namespace='SLP_model'):
        """
        Train the model on a chosen set of features. If none are chosen, the default is to re run the model with the current best_features attribute. Note that the training is carried out on the training data, X_train, only. To access the result, use:

        Args:
            features : list - train model with list of desired features
            epochs : int - number of epochs to train for
            units : int - number of units in hidden layer
            activation : str - choose activation function from keras options e.g. 'relu', 'sigmoid' etc.
            dropout : float - dropout rate from hidden layer to output
            batch_size : int - batch size for training
            save : bool - choose whether to save the model after training (This is safer than using the self.get_model() command)
            load : bool - choose whether to load the model from a saved instance
            namespace : str - file name space for saving/loading, will generate/look for files with this namespace and extensions .json/.h5

        Returns:

        Note: The input data for the SLP should be normalised such that each feature lies in the same range e.g. [0,1] to ensure that no one feature dominates. Non-normalised data will result in potentially very poor results.
        """
        if not isinstance(self.get_data('X_train'), pd.DataFrame):
            raise TypeError("ERROR: The input training data was not in the form of a pd.DataFrame.")
        print("Training - Single Layer Perceptron")
        print("==================================")
        print(" ")
        print("Running SLP regression classifier on feauture set:")
        print(" ")
        if features == None:
            features = self.get_best_features()
        pprint.pprint(features)
        print(" ")
        self._current_features = features
        X_train_data = self.get_data('X_train')
        X_val_data = self.get_data('X_val')
        X_test_data = self.get_data('X_test')
        Y_train_data = self.get_data('Y_train')
        Y_val_data = self.get_data('Y_val')
        Y_test_data = self.get_data('Y_test')
        X_train_data_temp = X_train_data[features]
        X_val_data_temp = X_val_data[features]
        X_test_data_temp = X_test_data[features]
        try:
            if not load:
                epochs=epochs
                input_dim = len(X_train_data_temp.columns)
                nn_model = Sequential()
                nn_model.add(Dense(units=units, activation=activation, input_dim=input_dim))
                nn_model.add(Dropout(dropout))
                nn_model.add(Dense(units=1))
                nn_model.compile(loss='mean_squared_error', optimizer='adam')
                history = nn_model.fit(X_train_data_temp, Y_train_data, epochs=epochs, verbose=1, batch_size=batch_size, validation_data=(X_val_data_temp, Y_val_data), shuffle=False)
                self.history = history.history
                if save:
                    temp_fname = namespace + '.txt'
                    history_file = open(temp_fname, 'wb')
                    pickle.dump(history.history, history_file)
                    history_file.close()
                    temp_fname = namespace + '.json'
                    json_file = open(temp_fname, 'w')
                    json_file.write(nn_model.to_json())
                    json_file.close()
                    temp_fname = namespace + '.h5'
                    nn_model.save_weights(temp_fname)
                    print("Saved Model to namespace: ", namespace)
            else:
                try:
                    temp_fname = namespace + '.json'
                    json_file = open(temp_fname, 'r')
                    nn_model = model_from_json(json_file.read())
                    json_file.close()
                    temp_fname = namespace + '.h5'
                    nn_model.load_weights(temp_fname)
                    temp_fname = namespace + '.txt'
                    history_file = open(temp_fname, 'rb')
                    history = pickle.load(history_file)
                    history_file.close()
                    self.history = history
                    print("Loaded Model from namespace: ", namespace)
                except (OSError, IOError) as e:
                    print("ERROR: Model not found.")
                    raise RuntimeError("Now exiting training.")
            self._model = nn_model
            Y_val_pred = nn_out_to_list(nn_model.predict(X_val_data_temp))
            Y_test_pred = nn_out_to_list(nn_model.predict(X_test_data_temp))
            final_rmse_val  = _test_metric(Y_val_data, Y_val_pred, 'rmse')
            self._val_rmse = final_rmse_val
            final_rmse_test = _test_metric(Y_test_data, Y_test_pred, 'rmse')
            print(' ')
            print('The RMSE on the validation set was: ', final_rmse_val[0])
            print('The mean percentage error is: ', final_rmse_val[1], '%.')
            print(' ')
            print('The RMSE on the test set was: ', final_rmse_test[0])
            print('The mean percentage error is: ', final_rmse_test[1], '%.')
            print('\nFinished training. To access the most recent classifier, call get_model(). To access the training history, use get_history().')
        except RuntimeError as re:
            print(re.args[0])

    def get_model(self):
        """
        Gets latest instance of SVR model as an instance of the sklearn.

        Returns:
            sklearn.svm.SVR - latest model instance from training or testing
        """
        if self._model != None:
            return self._model
        else:
            print("ERROR: No model has been trained. Please train a model.")

    def test_error(self, test_function='rmse'):
        """
        Computes the error in the current model with the given metric function. The result is stored in a class variable tget_testest_error.

        Args:
            test_function : string - for options see _test_metric?
        """
        Y_test_data = self.get_data('Y_test')
        X_test_data = self.get_data('X_test')
        Y_test_forecast = nn_out_to_list(self._model.predict(X_test_data[self._current_features]))
        test_rmse = _test_metric(Y_test_data, Y_test_forecast, test_function)
        self._test_error = test_rmse
        print('The RMSE on the test set was: ', test_rmse[0])
        print('The mean percentage error is: ', test_rmse[1], '%.')
        print('\nTo access the results, call get_test_error()')

    def get_test_error(self):
        """
        Returns most recent test error calculation results

        Retunrs:
            test_error : tuple - RMSE error and mean percentage error on test data
        """
        if self._test_error == None:
            return self.test_error()
        else:
            return self._test_error

    def get_history(self):
        """
        Returns model training history from most recent training session.

        Returns:
            history : dictionary - keras history data structure
        """
        return self.history

    def plot_training(self, save=False, save_name=None, _inline=False):
        """
        Plots the evolution of the training and validation error as a function of the number of epochs.

        Args:
            save : bool - defines whether the figure is saved or just shown
            save_name : string - filename if save is set to True
        """
        try:
            if self.history != None:
                plt.figure(figsize=(15,8))
                ax = plt.subplot()
                plt.plot(self.history['loss'], label = 'Training Loss')
                plt.plot(self.history['val_loss'], color='red', label = 'Validation Loss')
                plt.legend()
                x_label_setter('Epoch', plt.gca())
                y_label_setter('Loss', plt.gca())
            else:
                raise ValueError("ERROR: No model trained. Please train a model.")

            if not _inline:
                if (save_name == None and save):
                    save_string = 'SLP_forecast' + str(self.save_count) + '.png'
                    self.save_count += 1
                else:
                    save_string = save_name

                if save:
                    plt.savefig(save_string)
                else:
                    plt.show()
        except ValueError as ve:
            print(ve.args[0])

    def plot_forecast(self, x_data=["test"], save=False, save_name=None, _inline=False):
        """
        Plots the selected subset of data along with a pre selected confidence interval, options to save the figure with a designated filename are available. If no filename is given, the classifier will generate one.

        Args:
            x_data : list - default is test data, can choose subset from ["train", "test", "val"]
            save : bool - defines whether the figure is saved or just shown
            save_name : string - filename if save is set to True
        """
        data = self.get_data()
        temp_X = []
        temp_Y = []
        for _type in x_data:
            temp_X.append(data['X_' + _type])
            temp_Y.append(data['Y_' + _type])
        X_data = pd.concat(temp_X)
        Y_data = pd.concat(temp_Y)
        try:
            if self._model != None:
                X_data = X_data[self._current_features]
                Y_forecast  = nn_out_to_list(self._model.predict(X_data))
                # Y_data['Y_forecast'] = Y_forecast
                # Y_forecast_df = Y_data['Y_forecast']
                temp_df = pd.DataFrame({'Y_act': Y_data})
                temp_df['Y_forecast'] = Y_forecast
                plt.figure(figsize=(15,8))
                ax = plt.subplot()
                plt.plot('Y_act', data = temp_df, color = 'red', label = 'Actuals')
                plt.plot('Y_forecast', data = temp_df, color = 'blue', label = 'Prediction')
                plt.legend()
                x_label_setter('Date', plt.gca())
                y_label_setter('Data Value', plt.gca())
            else:
                raise ValueError("ERROR: No model trained. Please train a model.")

            if not _inline:
                if (save_name == None and save):
                    save_string = 'SLP_forecast' + str(self.save_count) + '.png'
                    self.save_count += 1
                else:
                    save_string = save_name

                if save:
                    plt.savefig(save_string)
                else:
                    plt.show()
        except ValueError as ve:
            print(ve.args[0])

    def plot_residuals(self, save=False, save_name=None, inline=False):
        """
        Plots the residuals against the input data to assess the distribution, options to save the figure with a designated filename are available. If no filename is given, the classifier will generate one.

        Args:
            x_data : list - default is test data, can choose subset from ["train", "test", "val"]
            conf_level : float - confidence level in the coefficients is extracted from the regression results
            save : bool - defines whether the figure is saved or just shown
            save_name : string - filename if save is set to True
        """
        try:
            if self._model == None:
                raise RuntimeError("No model trained, can't extract residuals.")
            fig = plt.figure(figsize=(10,10))
            residuals = self.get_data(data = 'Y_train') - nn_out_to_list(self._model.predict(self.get_data(data = 'X_train')[self._current_features]))
            Y_train_data = self.get_data(data='Y_train')
            sns.regplot(Y_train_data, residuals, lowess=True, line_kws={'color': 'red', 'lw': 3, 'alpha': 0.8})
            x_label_setter('y training values', plt.gca())
            y_label_setter('Residuals', plt.gca())

            if not inline:
                if (save_name == None and save):
                    save_string = 'SLP_residuals' + str(self.save_count) + '.png'
                    self.save_count += 1
                else:
                    save_string = save_name

                if save:
                    plt.savefig(save_string)
                else:
                    plt.show()
        except RuntimeError as re:
            print(re.args[0])

    def plot_autocorrelation(self, save=False, save_name=None, inline=False):
        """
        Plots the autocorrelation of the residuals, options to save the figure with a designated filename are available. If no filename is given, the classifier will generate one.

        Args:
            x_data : list - default is test data, can choose subset from ["train", "test", "val"]
            conf_level : float - confidence level in the coefficients is extracted from the regression results
            save : bool - defines whether the figure is saved or just shown
            save_name : string - filename if save is set to True
        """
        try:
            if self._model == None:
                raise RuntimeError("No model trained. Please train a model.")
            fig = plt.figure(figsize=(15, 6))
            residuals = self.get_data(data = 'Y_train') - nn_out_to_list(self._model.predict(self.get_data(data = 'X_train')[self._current_features]))
            autocorrelation_plot(residuals)
            x_label_setter('Lag', plt.gca())
            y_label_setter('Autocorrelation', plt.gca())

            if not inline:
                if (save_name == None and save):
                    save_string = 'SLP_autocorrelation' + str(self.save_count) + '.png'
                    self.save_count += 1
                else:
                    save_string = save_name

                if save:
                    plt.savefig(save_string)
                else:
                    plt.show()
        except RuntimeError as re:
            print(re.args[0])

    def help():
        helper_string = """
-----------------------------Help: XGBoost Regression Classifier-----------------------------

This is a XGBoost regression classifier for single variable output data.

To initialise a classifier, simply provide a feature set which matches the data you will use to train and test your model:

    clf = RFR(['time', 'temperature', ...])

For input data (X, Y), we can generate a training, validation, and testing split via:

    clf.generate_split(X, Y, train_size=0.7, val_size=0.15, test_size=0.15)

Alternatively, the user can generate their own data and then set it using:

    clf.set_data({'X_train' : X_train, 'Y_train' : Y_train, ...})

The data can then be accessed using:

    clf.get_data()
    or
    clf.get_data('X_train') etc.

There are a number of hyperparameters to optimise when it comes to the RFR, as such a pragmatic solution is to optimise the parameters with the full feature set and use this moving forward. To do this, perform a grid search using;

    clf.optimise_parameters()

The latest parameters from the optimisation can then be found using:

    clf.get_latest_params()

To perform the cross-validation for feature selection, run;

    clf.feature_selection()

This will make the classifiers best_features equal to this set.

To train the classifier, call:

    clf.train(features, C=100, epsilon=1.0, kernel='rbf', gamma=0.1, degree=2)

This generates an instance of an sklearn.svm.SVR model which is saved to a member variable and can be accesed using:

    clf.get_model()

The error on the test set can be generated from the latest parameters and accesed using the commands:

    clf.test_error()
    clf.get_test_error()

The data can be plotted either manually by the user or using the inbuilt:

    clf.plot_forecast(['train', 'test', ...])
    clf.plot_residuals()
    clf.plot_autocorrelation()
"""

        print(helper_string)
