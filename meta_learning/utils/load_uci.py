import numpy as np
import tensorflow as tf

def load_uci_data(datapath, dataset, merge = False, normalize_y = False):
    path = datapath + dataset + '/'
    data = np.loadtxt(path + 'data.txt')
    index_features = np.loadtxt(path + 'index_features.txt')
    index_target = np.loadtxt(path + 'index_target.txt')

    X = data[ : , np.array(index_features.tolist(), dtype=int) ]
    y = data[ : , np.array(index_target.tolist(), dtype=int) ]
    if dataset in ['colon', 'madelon']:
        y = (y + 1.0) / 2.0
    else:
        y = y - 1.0
    y = np.array(y, ndmin = 2).reshape((-1, 1))
    X = np.array(X, dtype='f')
    y = np.array(y, dtype='f')
    
    if merge:
        std_X = np.std(X, 0)
        std_X[ std_X == 0 ] = 1
        mean_X = np.mean(X, 0)
        X = (X - mean_X) / std_X
        if normalize_y:
            std_y = np.std(y, 0)
            std_y[ std_y == 0] = 1
            mean_y = np.mean(y, 0)
            y = (y - mean_y) / std_y
            return X, y, mean_y, std_y       

        else:
            return X, y

    else:
        total_dev = []
        total_test = []
        y_mean_std = []
        N_train = int(X.shape[0] * 0.4) 
        for i in xrange(5):
            #index_train = np.loadtxt(datapath + "index_train_{}.txt".format(i))
            #index_test = np.loadtxt(datapath + "index_test_{}.txt".format(i))
            # load training and test data
            #X_train = X[ np.array(index_train.tolist(), dtype=int), ]
            #y_train = y[ np.array(index_train.tolist(), dtype=int), ]
            #X_test = X[ np.array(index_test.tolist(), dtype=int), ]
            #y_test = y[ np.array(index_test.tolist(), dtype=int), ]
            np.random.seed(i*100)
            ind = np.random.permutation(range(X.shape[0]))
            ind_train = ind[:N_train]; ind_test = ind[N_train:]
            X_train = X[ind_train]; y_train = y[ind_train]
            X_test = X[ind_test]; y_test = y[ind_test]

            # We normalize the features
            std_X_train = np.std(X_train, 0)
            std_X_train[ std_X_train == 0 ] = 1
            mean_X_train = np.mean(X_train, 0)
            X_train = (X_train - mean_X_train) / std_X_train
            X_test = (X_test - mean_X_train) / std_X_train
            if normalize_y:
                std_y_train = np.std(y_train, 0)
                std_y_train[ std_y_train == 0 ] = 1
                mean_y_train = np.mean(y_train, 0)
                y_train = (y_train - mean_y_train) / std_y_train
                y_mean_std.append((mean_y_train, std_y_train))

            total_dev.append((X_train, y_train))
            total_test.append((X_test, y_test))

        if normalize_y:
            return total_dev, total_test, y_mean_std
        else: 
            return total_dev, total_test

