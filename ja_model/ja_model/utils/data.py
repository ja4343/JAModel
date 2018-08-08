from math import sqrt
import numpy as np
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from itertools import chain, combinations
from sklearn.metrics import mean_squared_error

def random_date_range(start_date, max_date, length=365):
    """
    random_date_range(start_date, max_date, length=365)

    Returns a random tuple of datetime strings with a date difference of (length) days from a given start date and a maximum end date.

    Args:
        start_date : datetime
        max_date : datetime - maximum end date
        length : int - length of datetime interval, default is one year
    Returns:
        st_date, en_date : string, string

    """
    day_difference_dt = max_date - start_date
    day_difference = day_difference_dt.days - 365
    random_days = random.randint(0, day_difference)
    start_date_dt = start_date + timedelta(days = random_days)
    end_date_dt = start_date_dt + timedelta(days = 365)
    st_date = start_date_dt.strftime("%d-%b-%Y")
    en_date = end_date_dt.strftime("%d-%b-%Y")
    return st_date, en_date

def powerset(iterable):
    """
    powerset([1,2,3]) --> [(), (1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)]

    Args:
        iterable : iterable (e.g. list, tuple,...) - set to generate possible subsets of

    Returns:
        list - list of possible subsets
    """
    xs = list(iterable)
    # note we return an iterator rather than a list
    tuples = chain.from_iterable(combinations(xs,n) for n in range(len(xs)+1))
    return list(tuples)

def _rmse(observations, forecast):
    """
    Args:
        observations : np.array - observed values in the test dataset
        forecast : np.array - predictions fronm the trained model
    Returns:
        (error, percent) : tuple - returns a tuple of the absolute error as well as the error as a percentage of the mean of the observed values
    """
    rmse = sqrt(mean_squared_error(observations, forecast))
    percent = 0
    for idx in range(len(observations)):
        percent += abs(observations[idx] - forecast[idx])/observations[idx]
    percent = 100*percent/len(observations)
    return (rmse, percent)

def _test_metric(observations, forecast, metric_function="rmse"):
    """
    Args:
        observations : np.array - observed values in the test dataset
        forecast : np.array - predictions fronm the trained model
        metric_function : string - choice of error metric function, availbale choices are ('rmse')
    Returns:
        (error, percent) : tuple - returns a tuple of the absolute error as well as the error as a percentage of the mean of the observed values
    """
    options = {'rmse': _rmse}
    options_string  = ''
    for key in options.keys():
        options_string = options_string + '"' + key + '"' + ', '
    options_string = options_string[:-2]

    try:
        if (len(observations) != len(forecast)):
            raise IndexError('ERROR: Observations and Forecast arrays are of different lengths.')
        else:
            temp_function = options[metric_function.lower()]
            return temp_function(observations, forecast)
    except IndexError as ie:
        print(ie.args[0])
    except KeyError as ke:
        print('ERROR: Available metric functions are ({})'.format(options_string))

if __name__ == "__main__":
    print('Inside data.py module, check path.')
