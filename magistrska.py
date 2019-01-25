import numpy as np
import holidays
from sklearn.metrics import mean_squared_error
from collections import defaultdict
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import json as json
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import VAR
from data_preparation import *
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import CuDNNLSTM
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from pandas import concat
from math import sqrt
from numpy import concatenate
import pickle
import pandas as pd
pd.set_option('display.expand_frame_repr', False)



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

def naive_forecast(data,train_size,time_horizons,location,from_date,to_date):

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
        results[time_frame] = sqrt(
            mean_squared_error(naive_forecast[first_test_date + pd.DateOffset(hours=time_frame):]["actual"].values,
                               naive_forecast[first_test_date + pd.DateOffset(hours=time_frame):][time_frame].values))

    #naive_forecast.plot()
    #plt.show()

    #time measurament
    results["time_elapsed"] = end - start

    return results,naive_forecast

def ARIMA_forecast(data,train_size,time_horizons,ARIMA_params,location,from_date,to_date):
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
        model_fit = model.fit(trend="nc", disp=False)
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
        results[time_frame] = sqrt(mean_squared_error(arima_forecast[first_test_date + pd.DateOffset(hours=time_frame):]["actual"].values,
                                                      arima_forecast[first_test_date + pd.DateOffset(hours=time_frame):][time_frame].values))

    # time measurament
    results["time_elapsed"] = end - start
    results["ARIMAParams"] = 'arima params={}'.format(ARIMAX_params)

    #arima_forecast.plot()
    #plt.show()

    return results,arima_forecast

def ARIMAX_forecast(data,train_size,time_horizons,ARIMAX_params,location,from_date,to_date):

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
        results[time_frame] = sqrt(
            mean_squared_error(arimax_forecast[first_test_date + pd.DateOffset(hours=time_frame):]["actual"].values,
                               arimax_forecast[first_test_date + pd.DateOffset(hours=time_frame):][time_frame].values))

    # time measurament
    results["time_elapsed"] = end - start
    results["ARIMAXParams"] = 'arimax params={}'.format(ARIMAX_params)

    #arimax_forecast.plot()
    #plt.show()
    return results, arimax_forecast

def VAR_forecast(data,train_size,features,time_horizons,VAR_params,location,from_date,to_date):
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
        model_fit = model.fit(1)
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
        results[time_frame] = sqrt(
            mean_squared_error(var_forecast[first_test_date + pd.DateOffset(hours=time_frame):]["actual"].values,
                               var_forecast[first_test_date + pd.DateOffset(hours=time_frame):][time_frame].values))

    # time measurament
    results["time_elapsed"] = end - start
    results["VARParams"] = 'var params={}'.format(VAR_params)

    #var_forecast.plot()
    #plt.show()
    return results, var_forecast

def LSTM_forecast(data,train_size,features,time_horizons,LSTM_params,location,from_date,to_date):

    train_data, test_data = train_test_split(data[features], train_size=train_size, shuffle=False)

    values = data[features].values
    values = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
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
        results[time_frame] = sqrt(
            mean_squared_error(lstm_forecast[first_test_date + pd.DateOffset(hours=time_frame):]["actual"].values,
                               lstm_forecast[first_test_date + pd.DateOffset(hours=time_frame):][time_frame].values))

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


def forecast(data,train_size,model,features,time_horizons,location,ARIMA_params,ARIMAX_params,VAR_params,LSTM_params,from_date,to_date):
    if (model == "naive"):
        naive_results,_ = naive_forecast(data=data,train_size=train_size,time_horizons=time_horizons,location=location,from_date=from_date,to_date=to_date)
        return naive_results
    elif (model == "ARIMA"):
        arima_results,_ = ARIMA_forecast(data=data,train_size=train_size,time_horizons=time_horizons,ARIMA_params=ARIMA_params,location=location,from_date=from_date,to_date=to_date)
        return arima_results
    elif (model == "ARIMAX"):
        arimax_results,_ = ARIMAX_forecast(data=data,train_size=train_size,time_horizons=time_horizons,ARIMAX_params=ARIMAX_params,location=location,from_date=from_date,to_date=to_date)
        return arimax_results
    elif (model == "VAR"):
        var_results,_ = VAR_forecast(data=data,train_size=train_size,features=features,time_horizons=time_horizons,VAR_params=VAR_params,location=location,from_date=from_date,to_date=to_date)
        return var_results
    elif (model == "LSTM"):
        lstm_reults,_ = LSTM_forecast(data=data,train_size=train_size,features=features,time_horizons=time_horizons,LSTM_params=LSTM_params,location=location,from_date=from_date,to_date=to_date)
        return lstm_reults

def master_thesis_method(data,models,features,train_size,time_horizons,ARIMA_params,ARIMAX_params,VAR_params,LSTM_params,location,from_date,to_date,time_interval):
    final_results = {}

    start = time.time()

    for model in models:
        model_results = forecast(data=data,train_size=train_size,model=model,features=features,time_horizons=time_horizons,
                                 ARIMA_params=ARIMA_params,ARIMAX_params=ARIMAX_params,VAR_params=VAR_params,LSTM_params=LSTM_params,
                                 location=location,from_date=from_date,to_date=to_date)
        final_results[model] = model_results
        print(model)
        print(model_results)

    end = time.time()
    total_time = "%.2f" % round(end-start,1)

    final_results["location"] = location
    final_results["from_date:to_date"] = str(from_date)+":"+str(to_date)
    final_results["train_size"] = train_size
    final_results["test_size"] = 1 - train_size
    final_results["time_interval"] = time_interval

    filename = 'loc={}_from_date={}_to_date={}_featu_used={}_t_size={}_interval={}_time={}s{}'.format(location,from_date,to_date,len(features),train_size,time_interval,total_time,".txt")

    df_results = DataFrame(final_results)

    #save results to disk

    f = open("results/"+filename,"w")
    f.write(df_results.to_string())
    f.close()


def data_preparation(data,missing_values,normalization):
    pass


##########################################
################ INPUTS ##################
##########################################

vosnjakova_bc = read_data_bc(4)

vosnjakova_bc.set_index('datetime',inplace=True)
vosnjakova = vosnjakova_bc[['bc']]
vosnjakova = vosnjakova.resample("H").mean().interpolate()

arso_weather = read_data_weather(8)
arso_weather.set_index('datetime',inplace=True)
arso_weather.drop(['location'],axis=1,inplace=True)
arso_weather = arso_weather.resample("H").mean().interpolate()

models = ["naive","ARIMA","ARIMAX","VAR","LSTM"]
time_horizons = [3,6,12,24,48]
from_date = "20180101"
to_date = "20180121"
train_size = 0.8
ARIMA_params = (1,0,1)
ARIMAX_params = (1,0,1)
VAR_params = (1,1,1)
#num_neurons,loss_function,optimizer,epochs,batch_size
LSTM_params = [50,"mae","adam",10,512]
data_vosnjakova = vosnjakova[from_date:to_date]
data_weather = arso_weather[from_date:to_date]
data_bleiweisova = read_data_traffic(location="1001-156").set_index('datetime').interpolate().resample("H").mean()[from_date:to_date]
data4 = pd.concat([data_bleiweisova,data_weather,data_vosnjakova], axis=1, ignore_index=False).drop(["location"],axis=1)
features = ["bc","pres","rh","humidity_mm","ws","wd","ws_max","glob_sev","dif_sev","temp"]
location = "Vosnjakova"

master_thesis_method(data=data4,models=models,features=features,time_horizons=time_horizons,train_size=train_size,
                     ARIMA_params=ARIMA_params,ARIMAX_params=ARIMAX_params,VAR_params=VAR_params,
                     LSTM_params=LSTM_params,location=location,from_date=from_date,to_date=to_date,time_interval="H")

#TODO: care about exogenous variables in ARIMAX, they should be future varibles for ARIMAX model to work correctly
#TODO: analyse trend, seasonality, ARIMAX, ARIMA, VAR, LSTM