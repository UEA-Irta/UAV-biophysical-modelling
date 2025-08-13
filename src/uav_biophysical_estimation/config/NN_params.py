
# NN_processing

NN_params = {
    'depth': 2,               # Number of hidden layers in the network
    'neurons': (64, 32),      # Number of neurons per hidden layer (tuple, one value for each layer)
    'activation': 'relu',     # Activation function to use ('relu', 'sigmoid', 'tanh', etc.)
    'optimizer': 'adam',      # Optimizer for training the model ('adam', 'sgd', etc.)
    'epochs': 100             # Number of epochs to train the model
}



NN_predict = {
    "dataset": "example_dataset_predict.xlsx",
    "features": ['fc','hc',	'hc_mean', 'NDVI_mean', 'NDVI_sd', 'MTVI2_mean', 'MTVI2_sd', 'SAVI_mean', 'SAVI_sd'],
    "target": 'LAI',
    "normalize": True,
    "min_max_values": None
}