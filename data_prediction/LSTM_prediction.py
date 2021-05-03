from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Dense
from keras import backend as K

from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import numpy as np

def vanila_LSTM(n_steps, n_features):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def stacked_LSTM(n_steps, n_features):
    # define model
    
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    """
    model = Sequential()
    model.add(LSTM(16, input_shape=(n_steps, n_features), return_sequences=True))
    model.add(LSTM(8, return_sequences=True))
    model.add(LSTM(4))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer=RMSprop())
    """
    return model

def bidirectional_LSTM(n_steps, n_features):
    # define model
    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def CNN_LSTM(n_steps, n_features):

    # define model
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

from keras.layers import ConvLSTM2D
def Conv_LSTM(n_steps, n_features, n_seq):
    # define model
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(n_seq, 1, n_steps, n_features)))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    return model

######

import matplotlib.pyplot as plt
def plot_loss(history, title):
    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(title)
    plt.xlabel('Nb Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    val_loss = history.history['val_loss']
    min_idx = np.argmin(val_loss)
    min_val_loss = val_loss[min_idx]
    print('Minimum validation loss of {} reached at epoch {}'.format(min_val_loss, min_idx))
def modelfit_save(model_name, train_model_file_name, n_steps, n_features, epochs_num, train, val, n_seq):
    K.clear_session()
    
    if model_name =='Vanila_LSTM':
        model = vanila_LSTM(n_steps, n_features)
    if model_name =='Stacked_LSTM':
        model = stacked_LSTM(n_steps, n_features)
    if model_name =='Biderectional_LSTM':
        model = bidirectional_LSTM(n_steps, n_features)
    if model_name == 'CNN_LSTM':
        model = CNN_LSTM(n_steps, n_features)
    if model_name == 'Conv_LSTM':
        model = Conv_LSTM(n_steps, n_features, n_seq)
        
    checkpointer, earlystopper = checkpointer_earlystopper(train_model_file_name, model)
    

    history = model.fit_generator(train
                                  , epochs=epochs_num
                                  , verbose=2
                                  , validation_data = val
                                 ,shuffle=False
                                , callbacks =[checkpointer, earlystopper])
    plot_loss(history, 'LSTM - Train & Test Loss')
    return model


from keras.callbacks import ModelCheckpoint, EarlyStopping
def checkpointer_earlystopper(model_name, model):
    #model_save
    json_name = model_name+'.json'
    h5_name = model_name+'.h5'
    model_json = model.to_json()
    with open(json_name, 'w') as json_file:
        json_file.write(model_json)        
    checkpointer = ModelCheckpoint(filepath=h5_name
                                   , verbose=2
                                   , save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss'
                                 , patience=10
                                 , verbose=2)
    
    return checkpointer, earlystopper

from keras.preprocessing.sequence import TimeseriesGenerator
def get_timeseries_generator(data_origin, target, n_past_steps, n_future_steps):
    data = data_origin.copy()
    data['target'] = data[target].shift(-(n_future_steps-1))
    data = data.dropna()
    dataset = data.drop('target', axis=1).values
    out_seq = data['target'].values
    generator = TimeseriesGenerator(dataset, out_seq, length=n_past_steps, batch_size=1, stride=1)
    print(len(generator))
    return generator


from keras.models import model_from_json   
from keras.optimizers import RMSprop
def load_model(load_model_name):
    json_file = open(load_model_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(load_model_name+".h5")
    print("Loaded model from disk")
    return loaded_model

from keras import losses
from sklearn.metrics import mean_squared_error
def eval_best_model(model_name, test):
    # Load model architecture from JSON
    best_model = load_model(model_name)
    # Compile the best model
    best_model.compile(loss=losses.mean_squared_error, optimizer=RMSprop())
    # Evaluate on test data
    perf_best_model = best_model.evaluate_generator(test)
    print('Loss on test data for {} : {}'.format(model_name, perf_best_model))
    
    return best_model

def get_rmse(best_model, test_generator, n_past_steps):
    y_pred = best_model.predict_generator(test_generator)
    
    index_ = n_past_steps
    y_original = test_generator.targets[index_:]
    
    plt.plot(y_pred, label='prediction')
    plt.plot(y_original, label='original')
    plt.legend()
    plt.show()
    
    rmse = np.sqrt(mean_squared_error(y_pred, y_original))
    print('RMSE: ', rmse)
    return rmse



def diff_scale_inverse_transform(offset, pred, scaler, station_flag):
    forecast = pred.copy()
    if 'scale' in station_flag:
        forecast = pd.DataFrame(scaler.inverse_transform(forecast), index=forecast.index,columns=forecast.columns)
    if 'diff' in station_flag:
        for col in pred.columns:
            forecast[str(col)]  = offset[col]+forecast[str(col)].cumsum()  
    return forecast