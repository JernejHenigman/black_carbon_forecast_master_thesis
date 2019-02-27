import numpy as np
import holidays
from sklearn.metrics import mean_squared_error
from collections import defaultdict
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import VAR
from data_preparation import *
from keras.models import Sequential
from keras.layers import CuDNNLSTM
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from pandas import concat
from math import sqrt
from numpy import concatenate
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)


#in this file we compare several ML models for time series blackcarbon concentration forecast.
#in particular we compare and gather results of the following models: naive method, ARIMA, ARIMAX, VAR, LSTM
#two of the testing models are univariate (naive, ARIMA), ARIMAX,VAR and LSTM are multivariate
#Each model is predicting black carbon concentration several hours ahead.
#more accurately, models are forecasting black carbon concentration 3,6,12,24,48 hours ahead
#we plot results and measure RMSE for each model for each time horizon forecast

#input: data --> Pandas DataFrame
#input: from_date --> string
#input: to_date --> string
#input: train_size --> int
#input: models --> list of strings
#input: time_horizons --> list of ints
#input: features --> list of feature strings

#output: RMSE for each time horizon
#output: time_of_execution

#each model returns RMSE score for each prediction (3,6,12,24,48 hours ahead)


def naive_forecast(data,train_size,time_horizons):

    #split data to train and test set
    train_data, test_data = train_test_split(data[['bc']], train_size=train_size,shuffle=False)
    #dict that holds predicted values
    predicted = defaultdict(list)
    #dict that holds results
    results = defaultdict(float)
    #list that holds actual values
    actual_values = []
    #list that holds index values

    first_test_date = test_data.iloc[0].name

    start = time.time()

    #main for loop that iterates over all test samples
    for i,test_value in enumerate(test_data.values[time_horizons[-1]:]):
        actual_values.append(test_value[0])

        predicted['index'].append(test_data.iloc[i].name)
        predicted['actual'].append(test_value[0])

        #loop that make prediction for each time horizion, usually [3,6,12,24,48]
        for time_frame in time_horizons:
            predicted[time_frame].append(test_data.values[i+time_horizons[-1]-time_frame-1][0])

    end = time.time()

    #we create pandas dataframe with all predictions stored in it
    naive_forecast = pd.DataFrame(predicted)
    naive_forecast.set_index('index', inplace=True)

    # shift results
    for time_frame in time_horizons:
        naive_forecast[time_frame] = naive_forecast[time_frame].shift(time_frame)

    # RMSE estimation for each time horizon
    for time_frame in time_horizons:
        all = sqrt(mean_squared_error(naive_forecast[first_test_date + pd.DateOffset(hours=time_frame):]["actual"].values,
                                      naive_forecast[first_test_date + pd.DateOffset(hours=time_frame):][time_frame].values))

        peak =  peak_estimation(naive_forecast[first_test_date + pd.DateOffset(hours=time_frame):][time_frame].values,
                                naive_forecast[first_test_date + pd.DateOffset(hours=time_frame):]["actual"].values)

        results[time_frame] = ("%.2f" % round(all,1),"%.2f" % round(peak,1))

    #naive_forecast.plot()
    #plt.show()

    #time measurament
    results["time_elapsed"] = end - start

    return results,naive_forecast

def ARIMA_forecast(data,train_size,time_horizons,ARIMA_params):
    # split data to train and test set
    train_data, test_data = train_test_split(data[['bc']], train_size=train_size, shuffle=False)
    # dict that holds predicted values
    predicted = defaultdict(list)
    # dict that holds results
    results = defaultdict(float)

    start = time.time()
    first_test_date = test_data.iloc[0].name

    #main for loop that iterates over all test samples
    for i,test_value in enumerate(test_data.values[:-time_horizons[-1]]):

        date = test_data.iloc[i].name

        #train ARIMA model
        model = ARIMA(train_data, order=ARIMA_params)
        model_fit = model.fit(trend="c", disp=False, transparams=False)
        arima_predictions = model_fit.predict(date, date + pd.DateOffset(hours=47))

        #each test row is added to train set in next iteration
        new_dataframe = pd.DataFrame({'datetime': date, 'bc': test_value[0]}, index=[0])
        new_dataframe = new_dataframe.set_index('datetime')
        train_data = train_data.append(new_dataframe)
        train_data.index = pd.DatetimeIndex(train_data.index.values,freq=train_data.index.inferred_freq)

        # we append datetime to dict that will serve as index
        predicted['index'].append(date)
        # we append actual value to dict
        predicted['actual'].append(test_value[0])
        #loop that get prediction for each time horizion, usually [3,6,12,24,48]

        for time_frame in time_horizons:
           predicted[time_frame].append(arima_predictions[time_frame-1])

    end = time.time()
    # dict to dataframe
    arima_forecast = pd.DataFrame(predicted)
    # we set index to datetime index
    arima_forecast.set_index('index', inplace=True)

    # shift results
    for time_frame in time_horizons:
        arima_forecast[time_frame] = arima_forecast[time_frame].shift(time_frame)

    # RMSE estimation for each time horizon
    for time_frame in time_horizons:
        all = sqrt(
            mean_squared_error(arima_forecast[first_test_date + pd.DateOffset(hours=time_frame):]["actual"].values,
                               arima_forecast[first_test_date + pd.DateOffset(hours=time_frame):][time_frame].values))

        peak = peak_estimation(arima_forecast[first_test_date + pd.DateOffset(hours=time_frame):][time_frame].values,
                               arima_forecast[first_test_date + pd.DateOffset(hours=time_frame):]["actual"].values)

        results[time_frame] = ("%.2f" % round(all, 1), "%.2f" % round(peak, 1))

    # time measurament
    results["time_elapsed"] = end - start
    results["ARIMAParams"] = 'arima params={}'.format(ARIMA_params)

    #arima_forecast.plot()
    #plt.show()

    return results,arima_forecast

def ARIMAX_forecast(data,train_size,time_horizons,ARIMAX_params):

    slo_holidays = holidays.CountryHoliday('SI')
    dates = data.index
    is_holiday = [1 if i in slo_holidays else 0 for i in dates.date]

    features = ['isHoliday']

    data['isHoliday'] = is_holiday

    # split data to train and test set
    train_y, test_y = train_test_split(data[['bc']], train_size=train_size, shuffle=False)
    train_x, test_x = train_test_split(data[['isHoliday']], train_size=train_size, shuffle=False)
    # dict that holds predicted values
    predicted = defaultdict(list)
    # dict that holds results
    results = defaultdict(float)

    start = time.time()
    first_test_date = test_y.iloc[0].name

    # main for loop that iterates over all test samples
    for i, test_value in enumerate(test_y.values[:-time_horizons[-1]]):

        date = test_y.iloc[i].name

        # train ARIMAX model
        model = ARIMA(endog=train_y, order=ARIMAX_params)
        model_fit = model.fit(disp=0)
        arimax_predictions = model_fit.predict(date, date + pd.DateOffset(hours=47),exog=train_x)

        # each test row is added to train set in next iteration
        new_dataframe_y = pd.DataFrame({'datetime': date, 'bc': test_value[0]}, index=[0])
        new_dataframe_y = new_dataframe_y.set_index('datetime')
        train_y = train_y.append(new_dataframe_y)
        train_y.index = pd.DatetimeIndex(train_y.index.values, freq=train_y.index.inferred_freq)

        feature_dict = defaultdict(float)
        feature_dict['datetime'] = date
        for j,feature in enumerate(features):
            feature_dict[feature] = test_x.iloc[i][j]

        # each test row is added to train feature set in next iteration
        new_dataframe_x = pd.DataFrame(feature_dict, index=[0])
        new_dataframe_x = new_dataframe_x.set_index('datetime')
        train_x = train_x.append(new_dataframe_x)
        train_x.index = pd.DatetimeIndex(train_x.index.values, freq=train_x.index.inferred_freq)

        # we append datetime to dict that will serve as index
        predicted['index'].append(date)
        # we append actual value to dict
        predicted['actual'].append(test_value[0])
        # loop that get prediction for each time horizion, usually [3,6,12,24,48]

        for time_frame in time_horizons:
            predicted[time_frame].append(arimax_predictions[time_frame - 1])

    end = time.time()
    # dict to dataframe
    arimax_forecast = pd.DataFrame(predicted)
    # we set index to datetime index
    arimax_forecast.set_index('index', inplace=True)

    # shift results
    for time_frame in time_horizons:
        arimax_forecast[time_frame] = arimax_forecast[time_frame].shift(time_frame)

    # RMSE estimation for each time horizon
    for time_frame in time_horizons:
        all = sqrt(
            mean_squared_error(arimax_forecast[first_test_date + pd.DateOffset(hours=time_frame):]["actual"].values,
                               arimax_forecast[first_test_date + pd.DateOffset(hours=time_frame):][time_frame].values))

        peak = peak_estimation(arimax_forecast[first_test_date + pd.DateOffset(hours=time_frame):][time_frame].values,
                               arimax_forecast[first_test_date + pd.DateOffset(hours=time_frame):]["actual"].values)

        results[time_frame] = ("%.2f" % round(all, 1), "%.2f" % round(peak, 1))

    # time measurament
    results["time_elapsed"] = end - start
    results["ARIMAXParams"] = 'arimax params={}'.format(ARIMAX_params)

    #arimax_forecast.plot()
    #plt.show()
    return results, arimax_forecast

def VAR_forecast(data,train_size,features,time_horizons,VAR_params):

    # split data to train and test set
    train_data, test_data = train_test_split(data[features], train_size=train_size, shuffle=False)
    # dict that holds predicted values
    predicted = defaultdict(list)
    # dict that holds results
    results = defaultdict(float)

    first_test_date = test_data.iloc[0].name
    start = time.time()

    # main for loop that iterates over all test samples
    for i, test_value in enumerate(test_data.values[:-time_horizons[-1]]):

        date = test_data.iloc[i].name

        model = VAR(train_data)
        model_fit = model.fit(VAR_params)
        var_predictions = model_fit.forecast(model_fit.y,steps=48)

        train_data = train_data.append(test_data.iloc[i])

        # we append datetime to dict that will serve as index
        predicted['index'].append(date)
        # we append actual bc value to dict
        predicted['actual'].append(test_value[0])

        # loop that get prediction for each time horizon, usually [3,6,12,24,48]
        for time_frame in time_horizons:
            predicted[time_frame].append(var_predictions[time_frame - 1][0])

    end = time.time()

    # dict to dataframe
    var_forecast = pd.DataFrame(predicted)

    # we set index to datetime index
    var_forecast.set_index('index', inplace=True)

    # shift results
    for time_frame in time_horizons:
        var_forecast[time_frame] = var_forecast[time_frame].shift(time_frame)

    # RMSE estimation for each time horizon
    for time_frame in time_horizons:
        all = sqrt(
            mean_squared_error(var_forecast[first_test_date + pd.DateOffset(hours=time_frame):]["actual"].values,
                               var_forecast[first_test_date + pd.DateOffset(hours=time_frame):][time_frame].values))

        peak = peak_estimation(var_forecast[first_test_date + pd.DateOffset(hours=time_frame):][time_frame].values,
                               var_forecast[first_test_date + pd.DateOffset(hours=time_frame):]["actual"].values)

        results[time_frame] = ("%.2f" % round(all, 1), "%.2f" % round(peak, 1))

    # time measurament
    results["time_elapsed"] = end - start
    results["VARParams"] = 'var params={}'.format(VAR_params)

    #var_forecast.plot()
    #plt.show()
    return results, var_forecast

def LSTM_forecast(data,train_size,features,time_horizons,LSTM_params):

    train_data, test_data = train_test_split(data[features], train_size=train_size, shuffle=False)

    values = data[features].values
    values = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(values)
    reframed = series_to_supervised(scaled, 1, 1)

    #we drop columns of other series at time t, only leave y variable, at time t
    reframed.drop(reframed.columns[np.arange(len(features) + 1, reframed.shape[1])], axis=1, inplace=True)

    values = reframed.values

    n_train_hours = int(reframed.shape[0]*train_size)
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]

    # design network
    model = Sequential()
    model.add(CuDNNLSTM(LSTM_params[0], input_shape=(1, len(features))))
    model.add(Dense(1))
    model.compile(loss=LSTM_params[1], optimizer=LSTM_params[2])

    # dict that holds predicted values
    predicted = defaultdict(list)

    # dict that holds results
    results = defaultdict(float)

    first_test_date = test_data.iloc[0].name

    i = 0
    start = time.time()

    while len(test) > 48:

        date = test_data.iloc[i].name

        # split into inputs and outputs
        train_X, train_y = train[:, :-1], train[:, -1]
        test_X, test_y = test[:48, :-1], test[:48, -1]

        # reshape input to be 3D sampled [samples,timestemps,features]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

        # fit network
        history = model.fit(train_X, train_y, epochs=LSTM_params[3], batch_size=LSTM_params[4], validation_data=(test_X, test_y), verbose=0, shuffle=False)

        # make a prediction
        yhat = model.predict(test_X)

        test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

        # invert scaling for forecast
        inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:, 0]

        # invert scaling for actual
        test_y = test_y.reshape((len(test_y), 1))
        inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:, 0]

        test_temp = test[0]
        #delete value from test
        test = np.delete(test,0,0)
        #add value to train
        train = np.concatenate((train,[test_temp]))

        # we append datetime to dict that will serve as index
        predicted['index'].append(date)
        # we append actual bc value to dict
        predicted['actual'].append(inv_y[0])

        # loop that get prediction for each time horizon, usually [3,6,12,24,48]
        for time_frame in time_horizons:
            predicted[time_frame].append(inv_yhat[time_frame - 1])

        i = i + 1

    end = time.time()

    lstm_forecast = pd.DataFrame(predicted)

    # we set index to datetime index
    lstm_forecast.set_index('index', inplace=True)

    # shift results
    for time_frame in time_horizons:
        lstm_forecast[time_frame] = lstm_forecast[time_frame].shift(time_frame)

    # RMSE estimation for each time horizon
    for time_frame in time_horizons:
        all = sqrt(
            mean_squared_error(lstm_forecast[first_test_date + pd.DateOffset(hours=time_frame):]["actual"].values,
                               lstm_forecast[first_test_date + pd.DateOffset(hours=time_frame):][time_frame].values))

        peak = peak_estimation(lstm_forecast[first_test_date + pd.DateOffset(hours=time_frame):][time_frame].values,
                               lstm_forecast[first_test_date + pd.DateOffset(hours=time_frame):]["actual"].values)

        results[time_frame] = ("%.2f" % round(all, 1), "%.2f" % round(peak, 1))

    # plot history
    #plt.plot(history.history['loss'], label='train')
    #plt.plot(history.history['val_loss'], label='test')
    #plt.legend()
    #plt.show()

    # time measurament
    results["time_elapsed"] = end - start
    results["LSTM_params"] = 'nodes={},loss={},opti={},epoch={},b_s={}'.format(LSTM_params[0],LSTM_params[1],LSTM_params[2],LSTM_params[3],LSTM_params[4])

    #lstm_forecast.plot()
    #plt.show()
    return results, lstm_forecast

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def forecast(data,train_size,model,features,time_horizons,ARIMA_params,ARIMAX_params,VAR_params,LSTM_params):
    if (model == "naive"):
        naive_results,_ = naive_forecast(data=data,train_size=train_size,time_horizons=time_horizons)
        return naive_results
    elif (model == "ARIMA"):
        arima_results,_ = ARIMA_forecast(data=data,train_size=train_size,time_horizons=time_horizons,ARIMA_params=ARIMA_params)
        return arima_results
    elif (model == "ARIMAX"):
        arimax_results,_ = ARIMAX_forecast(data=data,train_size=train_size,time_horizons=time_horizons,ARIMAX_params=ARIMAX_params)
        return arimax_results
    elif (model == "VAR"):
        var_results,_ = VAR_forecast(data=data,train_size=train_size,features=features,time_horizons=time_horizons,VAR_params=VAR_params)
        return var_results
    elif (model == "LSTM"):
        lstm_reults,_ = LSTM_forecast(data=data,train_size=train_size,features=features,time_horizons=time_horizons,LSTM_params=LSTM_params)
        return lstm_reults

def master_thesis_method(data,models,features,train_size,time_horizons,ARIMA_params,ARIMAX_params,VAR_params,LSTM_params,location,
                         from_date,to_date,time_interval,lagged_variables,derived_variables,background,micro_weather):
    final_results = {}

    start = time.time()

    for model in models:
        model_results = forecast(data=data,train_size=train_size,model=model,features=features,time_horizons=time_horizons,
                                 ARIMA_params=ARIMA_params,ARIMAX_params=ARIMAX_params,VAR_params=VAR_params,LSTM_params=LSTM_params)
        final_results[(model,"peak")] = model_results
        print(model)
        print(model_results)

    end = time.time()
    total_time = "%.2f" % round(end-start,1)

    final_results["location"] = location
    final_results["from_date:to_date"] = str(from_date)+":"+str(to_date)
    final_results["train_size"] = train_size
    final_results["test_size"] = 1 - train_size
    final_results["time_interval"] = time_interval
    final_results["lag_vars"] = str(lagged_variables)
    final_results["derived_vars"] = str(derived_variables)
    final_results["background"] = str(background)
    final_results["micro_weather"] = str(micro_weather)

    filename = 'loc={}_from_date={}_to_date={}_featu_used={}_t_size={}_interval={}_lag={}_derived_vars={}_background={}_micro_weather={}_time={}s{}'.format(
        location,from_date,to_date,len(features),train_size,time_interval,str(lagged_variables),str(derived_variables),str(background),micro_weather,total_time,".txt")

    df_results = DataFrame(final_results)

    #save results to disk

    f = open("results/"+filename,"w")
    f.write(df_results.to_string())
    f.close()

#method we use for defining how good model is in forcasting extreme (peak) values in time series data
def peak_estimation(yhat,actual):
    yhat_series = pd.Series(yhat).values
    actual_series = pd.Series(actual)
    #we define peak as values higher than 85 percentile
    actual_peaks = actual_series.where(actual_series > actual_series.quantile(q=0.85)).values

    #combine actual values and predicted values
    combined = list(zip(actual_peaks,yhat))
    #remove values that contains nan --> none peaks
    res = [i for i in combined if not pd.isna(i[0])]
    #split actual data and test data
    res = list(zip(*res))
    actual = list(res[0])
    predicted = list(res[1])

    #rmse of the model, only using peak values
    return sqrt(mean_squared_error(actual,predicted))

def data_preparation(bc_location,data_interval,date_from,date_to):

    data_bc = read_data_bc(bc_location)
    data_bc.set_index('datetime', inplace=True)
    data = data_bc[['bc']]
    data = data[date_from:date_to]
    data_black_carbon = data.resample(data_interval).mean().interpolate()

    arso_weather = read_data_weather(8)
    arso_weather.set_index('datetime', inplace=True)
    arso_weather.drop(['location'], axis=1, inplace=True)
    arso_weather = arso_weather[date_from:date_to]
    arso_weather_inter = arso_weather.resample(data_interval).mean().interpolate()

    pblh = read_data_pblh()
    pblh.set_index('datetime', inplace=True)
    pblh = pblh[date_from:date_to]
    pblh = pblh.resample(data_interval).mean().interpolate()

    location = "1001-156"

    #differnet traffic data, for differetn black carbon location
    if bc_location == 4:
        location = "1001-156"
    elif bc_location == 13:
        location = "1030-246"

    data_traffic = read_data_traffic(location=location).set_index('datetime').resample(data_interval).mean().interpolate()[from_date:to_date]

    extrapolated_traffic_data = extrapolate_traffic_data(data_traffic,date_to=to_date)

    data_combined = pd.concat([extrapolated_traffic_data, arso_weather_inter, data_black_carbon,pblh], axis=1, ignore_index=False).drop(["location"], axis=1)

    data_combined.fillna(0,inplace=True)

    return data_combined

def feature_importance(data):

    array = data.values
    # split into input and output
    X = array[:, 1:]
    y = array[:, 0]

    # fit random forest model
    model = RandomForestRegressor(n_estimators=500, random_state=1)
    model.fit(X, y)
    # show importance scores
    #print(model.feature_importances_)

    feature_importance = [(i,importance) for i,importance in enumerate(model.feature_importances_)]

    feature_importance_index = sorted(feature_importance,key=lambda x : x[1],reverse=True)

    sorted_names = [i[0] for i in feature_importance_index]
    sorted_importances = [i[1] for i in sorted(feature_importance,key=lambda x : x[1],reverse=True)]

    # plot importance scores
    sorted_names = data.columns.values[1:][sorted_names]
    names = data.columns.values[1:]
    ticks = [i for i in range(len(names))]

    print(sorted_names)
    print(sorted_importances)

    #plt.bar(ticks, sorted_importances)
    #plt.xticks(ticks, sorted_names)
    #plt.show()

def feature_selection(data):

    # separate into input and output variables
    array = data.values
    X = array[:, 1:]
    y = array[:, 0]
    # perform feature selection
    rfe = RFE(RandomForestRegressor(n_estimators=500, random_state=1), 5)
    fit = rfe.fit(X, y)
    # report selected features
    print('Selected Features:')
    names = data.columns.values[1:]
    for i in range(len(fit.support_)):
        if fit.support_[i]:
            print(names[i])
    # plot feature rank
    names = data.columns.values[1:]
    ticks = [i for i in range(len(names))]
    plt.bar(ticks, fit.ranking_)
    plt.xticks(ticks, names)
    plt.show()

def grid_serach_optimal_parameters(data,train_size,time_horizons):

    best_params = [(0,0,0) for i in range(len(time_horizons))]
    best_score = [100000000 for i in range(len(time_horizons))]

    for p in range(1,5):
        for d in range(0,1):
            for q in range(0,10):
                params = (p,d,q)
                try:
                    arima_results, _ = ARIMAX_forecast(data=data, train_size=train_size, time_horizons=time_horizons,ARIMAX_params=params)
                    print(arima_results)

                    for i,key in enumerate(list(arima_results.keys())[:5]):
                        if best_score[i] > float(arima_results[key][0]):
                            best_score[i] = float(arima_results[key][0])
                            best_params[i] = params

                            print("Best params"+str(time_horizons[i])+": "+str(params))

                except Exception as e:

                    print(e)


    print(best_params)

def grid_search_optimal_parameters_LSTM(data,train_size,features,time_horizons):

    best_params = [0,"mae","adam",0,0]
    best_score = [100000000 for i in range(len(time_horizons))]

    for neuron in range(25,200,25):
        for epoch in range(1,200,50):
            for batch_size in range(0,2048,128):

                LSTM_params = [neuron, "mae", "adam", epoch, batch_size]
                try:
                    lstm_reults, _ = LSTM_forecast(data, train_size, features, time_horizons, LSTM_params)

                    print(lstm_reults)

                    for i, key in enumerate(list(lstm_reults.keys())[:5]):
                        if best_score[i] > float(lstm_reults[key][0]):
                            best_score[i] = float(lstm_reults[key][0])
                            best_params[i] = LSTM_params

                            print("Best params" + str(time_horizons[i]) + ": " + str(best_params))

                except Exception as e:
                    print(e)

    print(best_params)

def grid_search_VAR_params(data,train_size,features,time_horizons):

    best_params = [0 for i in range(len(time_horizons))]
    best_score = [100000000 for i in range(len(time_horizons))]

    for parameter in range(1,15):
        var_results, _ = VAR_forecast(data=data, train_size=train_size,features=features,time_horizons=time_horizons,
                                      VAR_params=parameter)

        for j, key in enumerate(list(var_results.keys())[:5]):
            if best_score[j] > float(var_results[key][0]):
                best_score[j] = float(var_results[key][0])
                best_params[j] = parameter

                print("Best params" + str(time_horizons[j]) + ": " + str(parameter))
        print(var_results)


    print(best_params)

def spot_missing_data(data,horizon):

    print(len(data))
    for i in range(len(data.values)-1):

        diff =  data.iloc[i+1].name - data.iloc[i].name
        min_diff = divmod(diff.days * 86400 + diff.seconds, 60)[0]

        if (min_diff > 86400):
            print('from={} to={}'.format(data.iloc[i].name,data.iloc[i+1].name))

def correlations(data):
    series = data[["bc"]]
    plot_acf(series,lags=150)
    plt.show()

    plot_pacf(series, lags=150)
    plt.show()

#with function lag_variables we add lag variables as new features to the data set
def lag_variables(data,lag,features):
    data_with_lagged_variables = data[features]
    new_features = features
    for obs in range(1,lag):
        data_with_lagged_variables["bc_" + str(obs)] = data_with_lagged_variables.bc.shift(obs)

        new_features.append("bc_" + str(obs))

    data_with_lagged_variables.fillna(0.00,inplace=True)

    return data_with_lagged_variables, new_features

#we derive new features of the target variable such as mean, variance, quartiles,... for the last n hours
def derive_variables(data,features,window_size):
    data_with_derived_variables = data[features]
    new_features = features

    data_with_derived_variables["bc_mean_"+str(window_size)] = data_with_derived_variables.bc.rolling(window_size).mean()
    data_with_derived_variables["bc_min_" + str(window_size)] = data_with_derived_variables.bc.rolling(window_size).min()
    data_with_derived_variables["bc_max_" + str(window_size)] = data_with_derived_variables.bc.rolling(window_size).max()
    data_with_derived_variables["bc_std_" + str(window_size)] = data_with_derived_variables.bc.rolling(window_size).std()

    new_features.append("bc_mean_"+str(window_size))
    new_features.append("bc_min_"+str(window_size))
    new_features.append("bc_max_"+str(window_size))
    new_features.append("bc_std_"+str(window_size))

    data_with_derived_variables.fillna(0.00, inplace=True)

    return data_with_derived_variables,new_features

#With add_background function we manipulate input data. We do not use black carbon values from actual measuring station, rather we use
#city background black carbon (bc) values. We replace actual data (bc column) with bc column from background data. With this experiment, we are
#trying to predict actual bc values without help of actual black carbon past data. With this technique, we are trying to generalize forecast, for
#any given location in the city, where the city background is present not only for the 5 stationary locations, where aethalometers are located.
#We still compare forecast resutls with ground truths of actual bc values.
def add_background(data,data_background,train_size):

    data_with_background = data

    train_number = int(len(data_background)*train_size)

    backgroud_column_part = data_background.bc.values[:train_number]
    actual_column_part = data.bc.values[train_number:]

    new_bc_column = np.concatenate((backgroud_column_part,actual_column_part))
    data_with_background['bc'] = new_bc_column

    return data_with_background

#add_weathere function is similar than add_background. We replace weather data, in particular wind speed and wind direction and we also add some
#other weather variables to the input data. We check if forecast results are any better than with ARSO weather
def add_weather(data,micro_weather,features):

    data_with_new_weather = data

    data_with_new_weather['ws_h_avg'] = micro_weather['ws_h_avg']
    data_with_new_weather['ws_h_max'] = micro_weather['ws_h_max']
    data_with_new_weather['ws_h_min'] = micro_weather['ws_h_min']
    data_with_new_weather['ws_v_avg'] = micro_weather['ws_v_avg']
    data_with_new_weather['ws_v_max'] = micro_weather['ws_v_max']
    data_with_new_weather['ws_v_min'] = micro_weather['ws_v_min']
    data_with_new_weather['wd_h_avg'] = micro_weather['wd_h_avg']

    new_features = features + ['ws_h_avg','ws_h_max','ws_h_min','ws_v_avg','ws_v_max','ws_v_min','wd_h_avg']

    return data_with_new_weather,new_features


#time series decomposition on trend, seasonality and residual
def decompose_timeseries(data):
    series = data["bc"]
    result = seasonal_decompose(series, model='multiplicative')
    result.plot()
    plt.show()

def ad_fuller_test(data):

    for name in data.columns.values:
        print(name)
        series = data[name]
        X = series.values

        try:
            result = adfuller(X)
            print('ADF Statistic: %f' % result[0])
            print('p-value: %f' % result[1])
            print('Critical Values:')
            for key, value in result[4].items():
                print('\t%s: %.3f' % (key, value))
        except Exception as e:
            print("type error: " + str(e))

def extrapolate_traffic_data(traffic_data,date_to):

    last_date = traffic_data.iloc[-1].name

    a = pd.Timestamp(last_date)
    b = pd.Timestamp(date_to)
    hours_to_add = int((b - a)/np.timedelta64(1,'D')*24)

    new_data = traffic_data.groupby([traffic_data.index.weekday,traffic_data.index.hour]).mean().values

    date = last_date

    columns = traffic_data.columns.values

    for i in range(0,hours_to_add):
        date = date + pd.DateOffset(hours=1)
        day = date.weekday()
        hour = date.hour
        data = new_data[day*24+hour]
        dictionary = dict(zip(columns, data))
        traffic_data = traffic_data.append(pd.DataFrame(dictionary,index=[date]))

    return traffic_data

##########################################
################ INPUTS ##################
##########################################

#3 -> Posta, Traffic
#4 -> Vosnjakova, Traffic
#8 -> Bezigrad, Urban Background
#9 -> Kamniska, Urban Background
#10 -> Golovec, Background
#13 -> Zaloška, Traffic

location = "Vosnjakova"
bc_location = 4
data_interval = "H"
from_date = '20180101'
to_date = '20180131'

#models ---> 'naive','ARIMA','ARIMAX','VAR','LSTM'
models = ["ARIMAX"]
time_horizons = [3,6,12,24,48]
train_size = 0.85
ARIMA_params = (1,0,4)
ARIMAX_params = (1,0,4)
VAR_params = 2
#num_neurons,loss_function,optimizer,epochs,batch_size
LSTM_params = [50,"mae","adam",40,256]
features1 = ["bc"]
weather_features = ["pres","ws","ws_max","temp","humidity_mm","rh","dif_sev","glob_sev","pblh"]
traffic_features = ['oa1','lt1','tp1','tpp1','vavg1','vmax1','suma1',
                    'oa2','lt2','tpp2','vmin2','vavg2','vmax2','suma2','suma3']

weather_selected = True
traffic_selected = True

if weather_selected:
    features1 = features1 + weather_features

if traffic_selected:
    features1 = features1 + traffic_features

lagged_variables = None
derived_variables = True
with_background = True
with_micro_weather = None

micro_weather = read_data_weather(13)
micro_weather.set_index('datetime', inplace=True)
micro_weather.drop(['location'], axis=1, inplace=True)
micro_weather = micro_weather[from_date:to_date]
micro_weather_inter = micro_weather.resample("H").mean().interpolate()

for loc in [3]:

    features = ['bc', 'pres', 'ws', 'ws_max', 'temp', 'humidity_mm', 'rh', 'dif_sev', 'glob_sev', 'pblh', 'oa1', 'lt1', 'tp1', 'tpp1', 'vavg1', 'vmax1', 'suma1', 'oa2', 'lt2', 'tpp2', 'vmin2', 'vavg2', 'vmax2', 'suma2', 'suma3']

    data_combined = data_preparation(bc_location=loc,data_interval=data_interval,date_from=from_date,date_to=to_date)
    data_combined = data_combined[features]

    data_background = data_preparation(bc_location=loc,data_interval=data_interval,date_from=from_date,date_to=to_date)
    data_background = data_background[features]

    if (lagged_variables):
        data_combined,features = lag_variables(data=data_combined,lag=5,features=features)

    if (derived_variables):
        data_combined, features = derive_variables(data=data_combined,features=features,window_size=8)

    if (with_background):
        data_combined = add_background(data=data_combined,data_background=data_background,train_size=train_size)

    if (with_micro_weather):
        data_combined,features = add_weather(data=data_combined,micro_weather=micro_weather_inter,features=features)


    print("location:"+str(loc))
    feature_importance(data_combined[features])

#correlations(data_combined)

#grid_serach_optimal_parameters(data = data_combined,train_size=train_size,time_horizons=time_horizons)
#grid_search_VAR_params(data_combined,train_size,features,time_horizons)
#ad_fuller_test(data_combined)
#grid_search_optimal_parameters_LSTM(data_combined,train_size,features,time_horizons)

master_thesis_method(data=data_combined,models=models,features=features,time_horizons=time_horizons,train_size=train_size,
                     ARIMA_params=ARIMA_params,ARIMAX_params=ARIMAX_params,VAR_params=VAR_params,
                     LSTM_params=LSTM_params,location=location,from_date=from_date,to_date=to_date,time_interval=data_interval,
                     lagged_variables=lagged_variables,derived_variables = derived_variables,background = with_background,micro_weather = with_micro_weather)


#TODO: seasonality
#TODO: acf,pacf,normalization except LSTM