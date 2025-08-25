
from processing.splits_processing import DatasetProcessor
from processing.RF_processing import RFProcessor
from processing.NN_processing import NNProcessor
from config.splits_params import splits_params
from config.RF_params import RF_params, RF_predict
from config.NN_params import NN_params, NN_predict

import os
os.chdir(r'R:\Tesis_CesarMinuesa\Processing')

def main(splits=True, RF=False, ANN=True):

    # SPLITS
    if splits:
        processorSplits = DatasetProcessor(splits_params)
        processorSplits.normalize_splits()
        combined_X, combined_y, min_max_values = processorSplits.save_splits()
        X_train = combined_X['train']
        y_train = combined_y['train']
        X_val = combined_X['val']
        y_val = combined_y['val']
        X_test = combined_X['test']
        y_test = combined_y['test']

    # RF
    if RF:
        processorRF = RFProcessor('trial1_RF', RF_params, RF_predict)
        modelRF = processorRF.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
        processorRF.test(X_test=X_test, y_test=y_test, model=modelRF)
        processorRF.predict(model=modelRF, min_max_values=min_max_values)


    # ANN
    if ANN:
        processorNN = NNProcessor('trial1_NN', NN_params, NN_predict)
        modelNN = processorNN.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
        processorNN.test(X_test=X_test, y_test=y_test, model=modelNN)
        processorNN.predict(model=modelNN, min_max_values=min_max_values)



if __name__ == '__main__':
    main(True, True, True)