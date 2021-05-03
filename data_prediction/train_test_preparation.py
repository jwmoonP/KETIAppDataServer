import data_manager.data_preprocessing as pre
import pandas as pd
import numpy as np
#inverse transformation

#General
def get_train_test_data(station_flag, data, test_length):
    scaler='no_scaler'
    X_train, X_test = data[:-test_length], data[-test_length:]
    X_train_0, X_test_0= X_train.iloc[0], X_test.iloc[0]
    if 'diff' in station_flag:
        X_train, X_test = X_train.diff().fillna(0), X_test.diff().fillna(0)
    if 'scale' in station_flag:
        DP = pre.data_transformation()
        scaler, X_train = DP.data_scaling(X_train)
        X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns) 
    return X_train, X_train_0, X_test, X_test_0, scaler

from keras.preprocessing.sequence import TimeseriesGenerator
def get_train_test_lstm_data(station_flag, train, test, target_feature, n_past_steps, n_future_steps):
    train_x = train[:-(n_future_steps-1)]
    train_y = train[[target_feature]].shift(-(n_future_steps-1)).dropna()

    test_x = test[:-(n_future_steps-1)]
    test_y = test[[target_feature]].shift(-(n_future_steps-1)).dropna()
    
    train_x0 = train_x.iloc[0]
    train_y0 = train_y.iloc[n_past_steps-1]
    test_x0 = test_x.iloc[0]
    test_y0 = test_y.iloc[n_past_steps-1]

    scaler_x='no_scaler'
    scaler_y='no_scaler'
    if 'diff' in station_flag:
        train_x, train_y = train_x.diff().fillna(0), train_y.diff().fillna(0)
        test_x, test_y = test_x.diff().fillna(0), test_y.diff().fillna(0)
        
    if 'scale' in station_flag:
        DP = pre.data_transformation()
        scaler_x, train_x = DP.data_scaling(train_x)
        scaler_y, train_y = DP.data_scaling(train_y)
        test_x = pd.DataFrame(scaler_x.transform(test_x), index=test_x.index, columns=test_x.columns) 
        test_y = pd.DataFrame(scaler_y.transform(test_y), index=test_y.index, columns=test_y.columns) 
    
    if 'log' in station_flag:
        train_x, train_y = np.log(1+train_x), np.log(1+train_y)
        test_x, test_y = np.log(1+test_x), np.log(1+test_y)
    
    train_generator = TimeseriesGenerator(train_x.values, train_y.values, length=n_past_steps, batch_size=1, stride=1)
    test_generator = TimeseriesGenerator(test_x.values, test_y.values, length=n_past_steps, batch_size=1, stride=1)
    return train_generator, test_generator, scaler_x, scaler_y, train_x0, train_y0, test_x0, test_y0, train_x, train_y, test_x, test_y


def diff_scale_inverse_transform(offset, pred, scaler, station_flag):
    if 'log' in station_flag:
        pred = np.exp(pred)
        pred = pred-1
        
    if 'scale' in station_flag:
        pred = scaler.inverse_transform(pred)
        
    if 'diff' in station_flag:
        pred = offset + pred.cumsum()
    
    return pred