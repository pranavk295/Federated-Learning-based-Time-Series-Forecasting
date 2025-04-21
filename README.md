# Federated-Learning-based-Time-Series-Forecasting
This project focuses on developing a time series forecasting algorithm utilizing Federated Learning to accurately forecast electricity usage patterns while ensuring that sensitive data remains secure.

## Features: 
- **Data Security**: Leveraging Federated Learning to enable decentralized model training, ensuring that raw data never leaves the local device.
- **Deep Learning Models**: Implemented sophisticated neural network architectures, including Recurrent Neural Networks (LSTM and GRU) and Convolutional Neural Networks (CNNs), to effectively analyze and predict electricity consumption trends.
- **Multiprocessing**: Utilises multiprocessing for distributed model training (Federated Learning)
- **Model Checkpointing**: Mechanism to save the best model weights during training in a Federated Learning Round.
- **Experimentation Feature:** 
  - Allows users to choose the model (LSTM, GRU, CNN, etc.)
  - Option to choose number of Fed Learning Rounds
  - Option to specify the model weights aggregation method (Average/Median/Weighted Average)
  - Ability to select the number of clients participating in a Federated Learning Round.
  - Clustering feature to select similar type of clients in a federated learning round (eg. industrial, residential ,etc.)


## License
This project is licensed under the MIT License. See the LICENSE file for more details.
