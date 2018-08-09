import numpy as np
import seaborn as sns
from math import sqrt
from sklearn.metrics import mean_squared_error
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
import pprint
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot

from ja_model.utils import _test_metric, _rmse, powerset, simpleaxis, x_label_setter, y_label_setter
from ja_model.models import Model

class LinReg(Model):
    def __init__(self, features, *args, **kwargs):
        """
        Args:
            features : list - list of desired features; these should match the headings of the training and testing data used later in the classification

        For more information, see LinReg.help()
        """
        super().__init__(*args, **kwargs)
        print("Initialising a linear regression classifier")
        print("===========================================")
        print(" ")
        print("Feature Set")
        print("-----------")
        print(" ")
        pprint.pprint(features)
        print(" ")
        self.input_features = features
        self.best_features = features
        self._current_features = features
        self.__result = None
        self._was_regularised = False
        self.save_count = 0

    def get_latest_params(self):
        """
        Returns the latest regression run paraneters, either from the train() method, or the feature_selection() one.

        Returns:
            parameters : pd.Series - dataframe of parameters
        """
        return self._params

    def feature_selection(self, test_function='rmse'):
        """
        Runs through a feature selection algorithm which enumerates the possible subsets of the input features and attempts to minimise the test_metric error on the validation set after training the classifier on the training data. Updates the self.best_features attribute which can then be used to run the full model on the training and test data. This is only really appropriate for a relatively small number of features.

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
                lin_model = sm.OLS(Y_train_data, X_train_data_temp)
                result = lin_model.fit()
                val_forecast = (X_val_data_temp*result.params).sum(axis=1)
                val_rmse = _test_metric(Y_val_data, val_forecast, test_function)[0]
                list_results.append(val_rmse)
                # train_rsquared = result.rsquared
                # list_results.append(train_rsquared)
                counter += 1
            print('-------------------Finished iterating through possible feature sets.-------------------\n')
            test_mse_df = pd.DataFrame({'test_mse': list_results})
            lowest_test_mse = test_mse_df.sort_values(['test_mse'])
            index = lowest_test_mse.index
            # test_rsquared_df = pd.DataFrame({'test_rsquared': list_results})
            # highest_test_rsquared = test_rsquared_df.sort_values(['test_rsquared'], ascending=False)
            # index = highest_test_rsquared.index
            X_train_data_temp = X_train_data[feature_dict[index[0]]]
            X_val_data_temp = X_val_data[feature_dict[index[0]]]
            lin_model = sm.OLS(Y_train_data, X_train_data_temp)
            result = lin_model.fit()
            val_forecast = (X_val_data_temp*result.params).sum(axis=1)
            final_rmse = _test_metric(Y_val_data, val_forecast, test_function)
            print('Lowest Error on validation set with feature set: ', feature_dict[index[0]], '\n\n')
            print('Set best_features attribute to this set. With this choice, the following regression results were obtained on the training data:\n\n')
            self.best_features = feature_dict[index[0]]
            self.__result = result
            self._params = result.params
            self._was_regularised = False
            print(result.summary(), '\n\n')
            print('The RMSE on the validation set was: ', final_rmse[0])
            print('The mean percentage error is: ', final_rmse[1], '%.')
            print('\nFinished feature selection. To see list of best_features, call get_best_features() on your classifier. To access the regression parameters, call get_latest_params()')

        except TypeError as te:
            print(te.args[0])

    def train(self, features=None, regularised=False, alpha=0.5, l1_wt=0.0):
        """
        Train the model on a chosen set of features. If none are chosen, the default is to re run the model with the current best_features attribute. Note that the training is carried out on the training data, X_train, only. To access the result, use:

            clf.get_result()

        Args:
            features : list - train model with list of desired features
            regularised : bool - if True, a regularised fit model is applied, note also that data normalisation is carried out to run this method
            alpha : float - penalty weight in the regularised case
            L1_wt : float - float between 0 and 1, if 0 the fit is a ridge fit (default), if 1, it is a lasso fit

        Returns:
            result : sm.OLS.fit() object containing regression results

        Note: This also updates the clf.get_latest_params() attribute.
        """
        if not isinstance(self.get_data('X_train'), pd.DataFrame):
            raise TypeError("ERROR: The input training data was not in the form of a pd.DataFrame.")
        print("Training - Linear Regression Classifier")
        print("=======================================")
        print(" ")
        print("Running linear regression classifier on feauture set:")
        print(" ")
        if features == None:
            features = self.get_best_features()
        pprint.pprint(features)
        print(" ")
        self._current_features = features
        X_train_data = self.get_data('X_train')
        X_test_data = self.get_data('X_test')
        Y_train_data = self.get_data('Y_train')
        Y_test_data = self.get_data('Y_test')
        X_train_data_temp = X_train_data[features]
        X_test_data_temp = X_test_data[features]
        if not regularised:
            print("Regularisation not employed.")
            print(" ")
            lin_model = sm.OLS(Y_train_data, X_train_data_temp)
            result = lin_model.fit()
            test_forecast = (X_test_data_temp*result.params).sum(axis=1)
            final_rmse = _test_metric(Y_test_data, test_forecast, "rmse")
            self.__result = result
            self._params = result.params
            print(result.summary(), '\n\n')
            print('The RMSE on the test set was: ', final_rmse[0])
            print('The mean percentage error is: ', final_rmse[1], '%.')
            print('\nFinished training. To see full set of results, call get_result(). To access the regression parameters, call get_latest_params()')
            self._was_regularised = False
        else:
            lin_model = sm.OLS(Y_train_data, X_train_data_temp)
            result = lin_model.fit_regularized(alpha=alpha, L1_wt = l1_wt)
            test_forecast = (X_test_data_temp*result.params).sum(axis=1)
            final_rmse = _test_metric(Y_test_data, test_forecast, "rmse")
            self.__result = None
            self._params = result.params
            print(' ')
            pprint.pprint(result.params)
            print('The RMSE on the validation set was: ', final_rmse[0])
            print('The mean percentage error is: ', final_rmse[1], '%.')
            print('\nFinished training. To see full set of results, call get_result(). To access the regression parameters, call get_latest_params()')
            self._was_regularised = True

    def get_result(self):
        """
        Returns the result of the latest training run with the given feature set. To see the features used, consult get_result().summary()

        Returns:
            result - sm.OLS.fit() object
        """
        return self.__result

    def test_error(self, test_function='rmse'):
        """
        Computes the error in the current model with the given metric function. The result is stored in a class variable test_error.

        Args:
            test_function : string - for options see _test_metric?
        """
        Y_test_data = self.get_data('Y_test')
        X_test_data = self.get_data('X_test')
        Y_test_forecast = (X_test_data*self.get_latest_params()).sum(axis=1)
        test_rmse = _test_metric(Y_test_data, Y_test_forecast, test_function)
        self._test_error = test_rmse
        print('The RMSE on the test set was: ', test_rmse[0])
        if self._was_regularised:
            print(' ')
            print('NOTE: The model was regularised.')
            print(' ')
        print('The mean percentage error is: ', test_rmse[1], '%.')
        print('\nTo access the results, call get_test_error()')

    def get_test_error(self):
        """
        Returns most recent test error calculation results

        Retunrs:
            test_error : tuple - RMSE error and mean percentage error on test data
        """
        return self._test_error

    def plot_forecast(self, x_data=["test"], conf_levl=0.05, save=False, save_name=None, inline=False):
        """
        Plots the selected subset of data along with a pre selected confidence interval, options to save the figure with a designated filename are available. If no filename is given, the classifier will generate one.

        Args:
            x_data : list - default is test data, can choose subset from ["train", "test", "val"]
            conf_level : float - confidence level in the coefficients is extracted from the regression results
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
        if self.__result == None:
            X_data = X_data[self._current_features]
            Y_forecast   = (X_data*self.get_latest_params()).sum(axis=1)
            plt.figure(figsize=(15,8))
            ax = plt.subplot()
            Y_data.plot(ax = ax, color = 'red', label = 'Actuals')
            Y_forecast.plot(ax = ax, color='blue', label = 'Prediction')
            plt.legend()
            x_label_setter('Date', plt.gca())
            y_label_setter('Data Value', plt.gca())
        else:
            conf_int = self.__result.conf_int(alpha=conf_levl)
            forecast_lo = (X_data*conf_int[0]).sum(axis=1)
            forecast_hi = (X_data*conf_int[1]).sum(axis=1)
            Y_forecast   = (X_data*self.get_latest_params()).sum(axis=1)
            plt.figure(figsize=(15,8))
            ax = plt.subplot()
            Y_data.plot(ax = ax, color = 'red', label = 'Actuals')
            Y_forecast.plot(ax = ax, color='blue', label = 'Prediction')
            plt.legend()
            x_label_setter('Date', plt.gca())
            y_label_setter('Data Value', plt.gca())
            plt.fill_between(forecast_lo.index, forecast_lo, forecast_hi, color='gray', alpha=0.4)

        if not inline:
            if (save_name == None and save):
                save_string = 'LinReg_forecast' + str(self.save_count) + '.png'
                self.save_count += 1
            else:
                save_string = save_name

            if save:
                plt.savefig(save_string)
            else:
                plt.show()

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
            if self._was_regularised:
                raise RuntimeError("Model was regularised, can't extract residuals.")
            fig = plt.figure(figsize=(10,10))
            residuals = self.get_result().resid
            Y_train_data = self.get_data(data='Y_train')
            sns.regplot(Y_train_data, residuals, lowess=True, line_kws={'color': 'red', 'lw': 3, 'alpha': 0.8})
            x_label_setter('y training values', plt.gca())
            y_label_setter('Residuals', plt.gca())

            if not inline:
                if (save_name == None and save):
                    save_string = 'LinReg_residuals' + str(self.save_count) + '.png'
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
            if self._was_regularised:
                raise RuntimeError("Model was regularised, can't extract residuals.")
            fig = plt.figure(figsize=(15, 6))
            result = self.get_result()
            autocorrelation_plot(result.resid)
            x_label_setter('Lag', plt.gca())
            y_label_setter('Autocorrelation', plt.gca())

            if not inline:
                if (save_name == None and save):
                    save_string = 'LinReg_autocorrelation' + str(self.save_count) + '.png'
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
-----------------------------Help: Linear Regression Classifier-----------------------------

This is a linear regression classifier for single variable output data.

To initialise a classifier, simply provide a feature set which matches the data you will use to train and test your model:

    clf = LinReg(['time', 'temperature', ...])

For input data (X, Y), we can generate a training, validation, and testing split via:

    clf.generate_split(X, Y, train_size=0.7, val_size=0.15, test_size=0.15)

Alternatively, the user can generate their own data and then set it using:

    clf.set_data({'X_train' : X_train, 'Y_train' : Y_train, ...})

The data can then be accessed using:

    clf.get_data()
    or
    clf.get_data('X_train') etc.

To run a feature selection algorithm which selects the features which minimise the error on the validation set run:

    clf.feature_selection()

These features will then be stored for classification and be used in the remainder of the testing. To see a list of these features call:

    clf.get_best_features()

The latest parameters from the regression can then be found using:

    clf.get_latest_params()

Once a convenient set of features has been chosen, the train() method can be invoked, with or without the regularisation flag:

    clf.train(features, regularised, alpha, l1_weight)

This stores the result in a member variable which can be accessed using:

    clf.get_result()

The error on the test set can be generated from the latest parameters and accesed using the commands:

    clf.test_error()
    clf.get_test_error()

The data can be plotted either manually by the user or using the inbuilt:

    clf.plot_forecast(['train', 'test', ...])
    clf.plot_residuals()
    clf.plot_autocorrelation()
"""

        print(helper_string)

# END LinReg class
