
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM,GRU
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.regularizers import l2


tf.random.set_seed(0)


weights=[]
res_alloc_df=pd.read_excel('smart_grid_data/SME and Residential allocations.xlsx')
res_alloc_df=res_alloc_df[['ID','Code']]
history=5
future=1

def get_data():
    df= pd.read_csv('smart_grid_data/File1.txt',delimiter=' ')
    id_list=df['SMID'].unique()
    return df,id_list

def avg_weights():

    mean_weights = list()
    for layer_weights in zip(*weights): 
        mean_weights.append(
            np.array([np.array(w).mean(axis=0) for w in zip(*layer_weights)]))
    return mean_weights

def median_weights():
    median_weights = list()
    
    for layer_weights in zip(*weights):
        median_layer_weights = np.median(layer_weights, axis=0)
        median_weights.append(median_layer_weights)
    
    return median_weights
def weighted_avg_weights(performances):
    weighted_mean_weights = list()
    
    for layer_weights in zip(*weights):
        weighted_layer_weights = np.zeros_like(layer_weights[0])
        
        for client_weights, performance in zip(layer_weights, performances):
            weighted_layer_weights += client_weights * performance
        
        weighted_layer_weights /= np.sum(performances)  
        weighted_mean_weights.append(weighted_layer_weights)
    
    return weighted_mean_weights

     


def create_model_lstm(history, future):       
    print('Creating LSTM model...')
    global_model = Sequential()
    global_model.add(tf.keras.Input(shape=(history, 1)))
    global_model.add(LSTM(128, return_sequences=True))  
    global_model.add(Dropout(0.2))  
    global_model.add(LSTM(64, return_sequences=False)) 
    global_model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))  
    global_model.add(Dense(future, activation='linear'))  
    
    global_model.compile(loss='mse', metrics=['mae','mse'],optimizer=Adam(learning_rate=0.001))
    return global_model

def create_model_GRU(history,future):       
        print('Creating GRU model....')
        global_model = Sequential()
        global_model.add(tf.keras.Input(shape=(history, 1)))
        global_model.add(GRU(128, return_sequences=True)) 
        global_model.add(Dropout(0.2)) 
        global_model.add(GRU(64, return_sequences=False))  
        global_model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
        global_model.add(Dense(future, activation='linear')) 
        global_model.compile(loss='mse',metrics=['mae','mse'], optimizer=Adam(learning_rate=0.001)) 
        return global_model

def create_model_cnn_lstm(history,future):       
        print('Creating CNN-LSTM model....')
        global_model = Sequential()
        global_model.add(tf.keras.Input(shape=(history, 1)))
        global_model.add(tf.keras.layers.Conv1D(filters=6, kernel_size=history, activation='relu'))
        global_model.add(LSTM(6, return_sequences=True,activation='relu'))
        global_model.add(LSTM(6, return_sequences=False,activation='relu'))
        global_model.add(Dense(future))
        global_model.compile(loss='mse',metrics=['mae','mse'], optimizer='adam')
        return global_model
def create_model_cnn_GRU(history,future):       
        print('Creating CNN-GRU model....')
        global_model = Sequential()
        global_model.add(tf.keras.Input(shape=(history, 1)))
        global_model.add(tf.keras.layers.Conv1D(filters=6, kernel_size=history, activation='relu'))
        global_model.add(GRU(6, return_sequences=True,activation='relu'))
        global_model.add(GRU(6, return_sequences=False,activation='relu'))
        global_model.add(Dense(future))
        global_model.compile(loss='mse',metrics=['mae','mse'], optimizer='adam')
        return global_model