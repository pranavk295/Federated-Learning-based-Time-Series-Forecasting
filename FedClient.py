import multiprocessing
import random
import shutil
import sys
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf

import atexit
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error

tf.random.set_seed(0)
import FedServer
                        

def extract_datetime(time_id):
    # helper function that extracts the date from a time id object 
    day_no= str(time_id)[0:3]
    day_no=int(day_no)
    slot_no=str(time_id)[3:5]
    slot_no=int(slot_no)
    def_date=datetime(year=2009,month=1,day=1)
    day_delta=timedelta(days=day_no-1)
    month=def_date+day_delta
    time_delta=timedelta(minutes=(slot_no-1)*30)
    month=month+time_delta
    return month.strftime("%m/%d/%Y %H:%M:%S")
def getMeterType(id):
    # helper function which extracts what type of smart meter the client is using
    query_result=FedServer.res_alloc_df.query(f'ID=={id}')
    meter_type= query_result.loc[:, 'Code']
    return meter_type.astype(int)

def create_client(id):
    # Creates a client dataset to train the model
    client=pd.DataFrame()
    client=pd.concat([client,df[df['SMID'] ==id]])
    client=client[['TimeID','Energy']]
    client['Time']=client['TimeID'].apply(extract_datetime)
    client=client[['Time','Energy']]
    client.set_index('Time',inplace=True)
    client.index = pd.to_datetime(client.index)
    client=client.sort_index()
    client.index=client.index.to_period(freq='30min')
    scaler = StandardScaler()
    client[['Energy']] = scaler.fit_transform(client[['Energy']])

    return client



class Client:
    def __init__(self,id):
        self.id=id
        self.df=create_client(id)
        self.model=None
        self.X_train=None
        self.y_train=None
        self.X_val=None
        self.y_val=None
        self.X_test=None
        self.y_test=None
    def Transform_dataset(self,window_size=5):
        # Transforms the dataset suitable for neural network training (feature extraction)
        df_transform = self.df.to_numpy()
        X = []
        y = []
        for i in range(len(df_transform)-window_size):
            row = [a for a in df_transform[i:i+window_size]]
            X.append(row)
            label = df_transform[i+window_size][0]
            y.append(label)
        return np.array(X), np.array(y)
   
    def get_client_id(self):
        return f"Client: {self.id}"


    def train_test_split(self):
        # splitting client data to training, validation and testing split
        X,y=self.Transform_dataset(FedServer.history)
        self.X_train = X[:int(len(X)*0.80)]
        self.y_train= y[:int(len(X)*0.80)]
        self.X_val = X[int(len(X)*0.80):int(len(X)*0.90)]
        self.y_val=y[int(len(X)*0.80):int(len(X)*0.90)]
        self.X_test=X[int(len(X)*0.90):]
        self.y_test = y[int(len(X)*0.90):]

    def train_model(self):
        # Trains the global model with the client data
        print(f"Training global model at {self.get_client_id()}'s data.....")
        cp1 = ModelCheckpoint(f'model{self.get_client_id()}/', save_best_only=True)
        self.model.fit(self.X_train,self.y_train,epochs=10,validation_data=(self.X_val,self.y_val),batch_size=64,verbose=0,callbacks=[cp1])
        self.model = load_model(f'model{self.get_client_id()}/')
        
    def evaluate(self,return_loss=False):
        # Evaluates global model performance on test data
        if return_loss:
            loss=self.model.evaluate(self.X_test,self.y_test,verbose=0)
            return loss
        start=0
        end=100
        print('Evalutating Model...')
        predictions = self.model.predict(self.X_test).flatten()
        new_df = pd.DataFrame(data={'Predictions':predictions, 'Actuals':self.y_test})
        plt.plot(new_df['Predictions'][start:end],color='red')
        plt.plot(new_df['Actuals'][start:end],color='blue')
        plt.show()
        self.model.evaluate(self.X_test,self.y_test)

def train_client_multiprocess(client,global_model,clients):
    # function helps to train multiple clients parallely using multiprocessing
    client.train_test_split()
    client.model = global_model
    client.train_model()
    clients.put(client)
    print(f"Training completed for {client.get_client_id()}")

def selectClients(client_list,no_of_clients,selection_method):
    # Function to help select clients for federated learning

    if selection_method.lower()=='residential':
        residential=[x for x in client_list if int(getMeterType(x))==1]
        # residential=residential[:no_of_clients]
        residential=random.sample(residential,no_of_clients)
        client_ids=residential
    elif selection_method.lower()=='industrial':
        industrial=[x for x in client_list if int(getMeterType(x))==2]
        # industrial=industrial[:no_of_clients]
        industrial=random.sample(industrial,no_of_clients)
        client_ids=industrial
    elif selection_method.lower()=='random':
        # rand_ids=random.sample(client_list.tolist(),no_of_clients)
        rand_ids=client_list[0:no_of_clients]
        client_ids=rand_ids
    else:
        rand_ids= [1890]
        client_ids=rand_ids

    return client_ids
    
    
# Process termination mechanism in case there is a  Keyboard Interrupt
def cleanup():
    print("Cleaning up processes...")
    for process in multiprocessing.active_children():
        print("Terminating process:", process.pid)
        process.terminate()
        process.join()


if __name__=='__main__':
    atexit.register(cleanup)
    df,id_list=FedServer.get_data()
    model=sys.argv[1]
    number_of_clients=int(sys.argv[2])
    number_of_rounds=int(sys.argv[3])
    sampling_method=sys.argv[4]
    agg_type=sys.argv[5]
    # global_model=None
    #Global model selection
    if model.lower()=='lstm':
        print('Global Model:','LSTM')
        global_model=FedServer.create_model_lstm(FedServer.history,FedServer.future)
    elif model.lower()=='gru':
        print('Global Model:','GRU')
        global_model=FedServer.create_model_GRU(FedServer.history,FedServer.future)
    elif model.lower()=='cnn_lstm':
        print('Global Model:','CNN-LSTM')
        global_model=FedServer.create_model_cnn_lstm(FedServer.history,FedServer.future)
    elif model.lower()=='cnn_gru':
        print('Global Model:','CNN-GRU')
        global_model=FedServer.create_model_cnn_GRU(FedServer.history,FedServer.future)
    else:
        print('Global Model:','CNN-LSTM')
        global_model=FedServer.create_model_cnn_lstm(FedServer.history,FedServer.future)

        
    j=0
    directories=[]
    # Selecting clients based on their meter type or randomly
    client_ids=selectClients(id_list,number_of_clients,sampling_method.lower())
    # Federated Learning Loop
    while j<number_of_rounds:
        print(f'\nRound {j+1}\n')
        print('Collecting weights from clients training the global model at their machine\n')
        processes=[]
        client_list=[]

        clients=multiprocessing.Manager().Queue()
        # multiprocessing for parallel training
        for i in client_ids:
            client = Client(i)
            directories.append(f'model{client.get_client_id()}/')
            process = multiprocessing.Process(target=train_client_multiprocess, args=(client,global_model,clients))
            processes.append(process)
            process.start()

        for process in processes:
            client_list.append(clients.get())
            process.join()
    
        # store loss of global model on each client's data. (Helpful for weighted averaging aggregation)
        losses=[]
        for client in client_list:
            losses.append(client.evaluate(return_loss=True)[0])
        
        # Store local model weights for each client
        for client in client_list:
            FedServer.weights.append(client.model.get_weights())

            
        print('\nClient side training complete...\n')

        print('Aggregating the weights returned from the clients')
        print('Updating global model weights with the aggregated weights')
        # Selecting federated aggregation technique.
        if agg_type.lower()=='avg':
            global_model.set_weights(FedServer.avg_weights())
        elif agg_type.lower()=='median':
            global_model.set_weights(FedServer.median_weights())
        elif agg_type.lower()=='wavg':
            client_performances=[1/(1+loss) for loss in losses]
            total_performance = sum(client_performances)
            client_performances = [performance / total_performance for performance in client_performances]
    
            global_model.set_weights(FedServer.weighted_avg_weights(client_performances))
        else:
            global_model.set_weights(FedServer.avg_weights())

        print('Sending new model to the clients and evaluating performance')
        for i in client_ids:
            client=Client(i)
            client.train_test_split()
            client.model=global_model
            client.evaluate()
        FedServer.weights.clear()
        j+=1
    for dir in set(directories):
        try:
            shutil.rmtree(dir)
            print(f"Removed directory: {dir}")
        except Exception as e:
            print(f"Error removing directory {dir}: {e}")
