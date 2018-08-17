import numpy as np
import seaborn as sns
from math import sqrt
from sklearn.metrics import mean_squared_error
import pandas as pd
import statsmodels.api as sm
import pprint
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

from ja_model.utils import _test_metric, _rmse, powerset, simpleaxis, x_label_setter, y_label_setter
from ja_model.models import Model

class ARIMAX(Model):
    def __init__(self, features, p=2, d=0, q=0, *args, **kwargs):
        """
        Args:
            features : list - list of desired features, these should match the headings of the training and testing data used later in the classification
            p : int - the lag for the AR(p) component of the ARIMA process
            d : int - the difference parameter
            q : int - the parameter in the MA(q) aspect of the process
        """
        super().__init__(*args, **kwargs)
        print("Initialising a ARIMAX Model (p = {}, d = {}, q = {})".format(p, d, q))
        print("=================================================")
        print(" ")
        print("Feature Set")
        print("-----------")
        print(" ")
        pprint.pprint(features)
        print(" ")
        print("IMPORTANT: When providing the feature set, the time series variable should not appear, either as a raw value, or as a lagged series. This is already included in the ARIMA component.")
        self.input_features = features
        self.best_features = features
        self._current_features = features
        self.p = p
        self.d = d
        self.q = q
        self.__result = None
        self._model = None
        self.save_count = 0

    def get_summary(self):
        """
        Returns the latest run hyperparaneters

        Returns:
            parameters : dictionary - dictionary of hyperparameters
        """
        if self._model != None:
            return self._model.summary()
        else:
            print("ERROR: No model has been trained. Please train a model.")

    def feature_selection(self, test_function='rmse'):
        """
        Runs through a feature selection algorithm which enumerates the possible subsets of the input features and attempts to minimise the test_metric error on the validation set after training the classifier on the training data. Updates the self.best_features attribute which can then be used to run the full model on the training and test data. This is only really appropriate for a relatively small number of features. To avoid computational intensity, there is no hyperparameter optimisation, instead standard parameters are calculated from the data. Once the best feature set has been identified, one can use the additional functionality in the library to tune the hyperparameters.

        Args:
            test_function : function - default is rmse testing, but others are available, see _test_metric? for more information

        Note: the input data for the features must be in the form of a pd.DataFrame
        """
        try:
            if not isinstance(self.get_data('X_train'), pd.DataFrame):
                raise TypeError("ERROR: The input training data was not in the form of a pd.DataFrame.")
            feature_set = list(powerset(self.input_features))
            print("Feature Selection")
            print("=================")
            print(" ")
            print("Running feature selection on a feature set of size: ", len(feature_set) - 1)
            print(" ")
            feature_dict = {}
            list_results = []
            counter = 0
            X_train_data = self.get_data('X_train')
            X_val_data = self.get_data('X_val')
            Y_train_data = self.get_data('Y_train')
            Y_val_data = self.get_data('Y_val')

            if (len(feature_set) < 100):
                counter_check = 10
            elif (len(feature_set) < 1000):
                counter_check = 100
            elif (len(feature_set) < 2500):
                counter_check = 250
            elif (len(feature_set) < 5000):
                counter_check = 500
            else:
                counter_check = 1000

            for _features in feature_set[1:]:
                if (counter % counter_check == counter_check - 1):
                        print('-------------------Completed ', counter + 1, ' feature sets out of ', len(feature_set) - 1, '-------------------\n')
                X_train_data_temp = X_train_data[list(_features)]
                X_val_data_temp = X_val_data[list(_features)]
                feature_dict[counter] = list(_features)
                temp_model = SARIMAX(endog=Y_train_data, exog = X_train_data_temp, order=(self.p, self.d, self.q))
                try:
                    temp_model_fit = temp_model.fit(disp=0)
                    val_forecast = temp_model_fit.forecast(len(Y_val_data), exog = np.array(X_val_data_temp).reshape(len(Y_val_data), len(X_val_data_temp.columns)))
                    val_rmse = _test_metric(Y_val_data, val_forecast, test_function)[0]
                except:
                    print("WARNING: ARIMAX did not converge for the feature set " + list(_feautures) + " this will not appear in the final analysis.")
                    val_rmse = 10000000
                list_results.append(val_rmse)
                counter += 1
            print('-------------------Finished iterating through possible feature sets.-------------------\n')
            test_mse_df = pd.DataFrame({'test_mse': list_results})
            lowest_test_mse = test_mse_df.sort_values(['test_mse'])
            index = lowest_test_mse.index
            self.best_features = feature_dict[index[0]]
            X_train_data_temp = X_train_data[feature_dict[index[0]]]
            X_val_data_temp = X_val_data[feature_dict[index[0]]]
            temp_model = SARIMAX(endog=Y_train_data, exog = X_train_data_temp, order=(self.p, self.d, self.q))
            temp_model_fit = temp_model.fit(disp=0)
            val_forecast = temp_model_fit.forecast(len(Y_val_data), exog = np.array(X_val_data_temp).reshape(len(Y_val_data), len(X_val_data_temp.columns)))
            val_rmse = _test_metric(Y_val_data, val_forecast, test_function)
            print('Lowest Error on validation set with feature set: ', feature_dict[index[0]], '\n\n')
            print('Set best_features attribute to this set. With this choice, the following regression results were obtained on the training data:\n\n')
            print('The RMSE on the validation set was: ', val_rmse[0])
            print('The mean percentage error is: ', val_rmse[1], '%.')
            print('\nFinished feature selection. To see list of best_features, call get_best_features() on your classifier. To access the regression parameters, call get_latest_params()')

        except TypeError as te:
            print(te.args[0])

    def train(self, features=None):
        """
        Train the model on a chosen set of features. If none are chosen, the default is to re run the model with the current best_features attribute. Note that the training is carried out on the training data, X_train, only. To access the result, use:

        Args:
            features : list - train model with list of desired features

        Returns:
        """
        if not isinstance(self.get_data('X_train'), pd.DataFrame):
            raise TypeError("ERROR: The input training data was not in the form of a pd.DataFrame.")
        print(' ')
        print("Training - ARIMAX")
        print("=================")
        print(" ")
        print("Running ARIMAX model on feauture set:")
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
        model = SARIMAX(endog=pd.concat([Y_train_data, Y_val_data]), exog=pd.concat([X_train_data_temp, X_val_data_temp]), order=(self.p,self.d,self.q))
        model_fit = model.fit(disp=0)
        self._model = model_fit
        Y_test_pred = model_fit.forecast(len(Y_test_data), exog = np.array(X_test_data_temp).reshape(len(Y_test_data), len(X_test_data_temp.columns)))
        final_rmse_test = _test_metric(Y_test_data, Y_test_pred, 'rmse')
        self._test_error = final_rmse_test
        print(' ')
        print('The RMSE on the test set was: ', final_rmse_test[0])
        print('The mean percentage error is: ', final_rmse_test[1], '%.')
        print('\nFinished training. To access the most recent classifier, call get_model()')

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

    def get_test_error(self):
        """
        Returns most recent test error calculation results

        Retunrs:
            test_error : tuple - RMSE error and mean percentage error on test data
        """
        return self._test_error

    def plot_forecast(self, save=False, save_name=None, _inline=False):
        """
        Plots the selected subset of data along with a pre selected confidence interval, options to save the figure with a designated filename are available. If no filename is given, the classifier will generate one.

        Args:
            save : bool - defines whether the figure is saved or just shown
            save_name : string - filename if save is set to True
        """
        x_data = ['test']
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
                Y_forecast  = self._model.forecast(len(Y_data), exog = np.array(X_data).reshape(len(Y_data), len(X_data.columns)))
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
                    save_string = 'ARIMAX' + str(self.save_count) + '.png'
                    self.save_count += 1
                else:
                    save_string = save_name

                if save:
                    plt.savefig(save_string)
                else:
                    plt.show()
        except ValueError as ve:
            print(ve.args[0])

    def plot_acf(self, lags=40, save=False, save_name=None, _inline=False):
        """
        Plots the autocorrelation function for the training data, this can be used to estimate the values of p and q which can then be set using the member variables

        Args:
            lags : int - choose how many lags to plot
            save : bool - defines whether the figure is saved or just shown
            save_name : string - filename if save is set to True
        """
        try:
            series = self.get_data('Y_train')
            plot_acf(series, lags=lags)
            if not _inline:
                if (save_name == None and save):
                    save_string = 'ACF' + str(self.save_count) + '.png'
                    self.save_count += 1
                else:
                    save_string = save_name

                if save:
                    plt.savefig(save_string)
                else:
                    plt.show()
        except:
            print("ERROR: Data Error, please check time series data.")

    def plot_pacf(self, lags=40, save=False, save_name=None, _inline=False):
        """
        Plots the autocorrelation function for the training data, this can be used to estimate the values of p and q which can then be set using the member variables

        Args:
            lags : int - choose how many lags to plot
            save : bool - defines whether the figure is saved or just shown
            save_name : string - filename if save is set to True
        """
        try:
            series = self.get_data('Y_train')
            plot_pacf(series, lags=lags)
            if not _inline:
                if (save_name == None and save):
                    save_string = 'ACF' + str(self.save_count) + '.png'
                    self.save_count += 1
                else:
                    save_string = save_name

                if save:
                    plt.savefig(save_string)
                else:
                    plt.show()
        except:
            print("ERROR: Data Error, please check time series data.")
