import numpy as np
import tensorflow as tf

def load_data(train_data_index):
    # load data
    from sklearn.datasets import load_svmlight_file
    name = 'data/a' + str(train_data_index)
    print 'loading %sa.txt as training data...' % name
    X_train_1, y_train_1 = load_svmlight_file(name + 'a.txt', n_features=123, dtype=np.float32)
    X_train_2, y_train_2 = load_svmlight_file(name + 'a.t', n_features=123, dtype=np.float32)
    X_train = np.concatenate([X_train_1.toarray(), X_train_2.toarray()], axis=0)
    y_train = np.concatenate([y_train_1, y_train_2], axis=0).reshape(-1, 1)
    y_train = (y_train + 1) / 2	# make it in {0, 1}

    total_dev = []
    total_test = []
    #np.random.seed(0)
    #ind = np.random.permutation(range(1, 10))
    for i in range(1, 10):
        if i != train_data_index:
            name_dev = 'data/a' + str(i)+ 'a.txt'
            name_test = 'data/a' + str(i) + 'a.t'
            print 'loading ' + name_dev + ' as test data...'
            X_dev, y_dev = load_svmlight_file(name_dev, n_features=123, dtype=np.float32)
            y_dev = (y_dev + 1) / 2
            X_test, y_test = load_svmlight_file(name_test, n_features=123, dtype=np.float32)
            y_test = (y_test + 1) / 2
            total_dev.append((X_dev.toarray(), y_dev.reshape(-1, 1)))
            total_test.append((X_test.toarray(), y_test.reshape(-1, 1)))

    return X_train, y_train, total_dev, total_test

def load_uci_data(datapath, dataset, merge = False):
    path = datapath + dataset + '/'
    data = np.loadtxt(path + 'data.txt')
    index_features = np.loadtxt(path + 'index_features.txt')
    index_target = np.loadtxt(path + 'index_target.txt')

    X = data[ : , np.array(index_features.tolist(), dtype=int) ]
    y = data[ : , np.array(index_target.tolist(), dtype=int) ]
    y = y - 1.0
    y = np.array(y, ndmin = 2).reshape((-1, 1))
    X = np.array(X, dtype='f')
    y = np.array(y, dtype='f')
    
    if merge:
        std_X = np.std(X, 0)
        std_X[ std_X == 0 ] = 1
        mean_X = np.mean(X, 0)
        X = (X - mean_X) / std_X
        
        return X, y

    else:
        total_dev = []
        total_test = []
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
            total_dev.append((X_train, y_train))
            total_test.append((X_test, y_test))
 
        return total_dev, total_test
    
def load_surf_data(train_data_name, seed = 0, ratio = 0.9):
    from scipy.io import loadmat
    from scipy.stats import zscore
    data_name = ['amazon', 'caltech']#, 'dslr']
    path = '/homes/mlghomes/yl494/proj/ideas_to_be_tested/TL/classification/data/data/' 
    
    def preprocessing(data):
        X = np.asarray(data['fts'], dtype='f')
        X = zscore(X / np.sum(X, 1, keepdims=True), axis=0)
        label = np.array(data['labels'] - 1, dtype='int')
        N = label.shape[0]; num_class = label.max() + 1
        y = np.zeros([N, num_class])
        y[np.arange(N), label.ravel()] = 1
        y = np.asarray(y, dtype='f')
        return X, y
    
    X_train, y_train = preprocessing(loadmat(path + train_data_name + '_SURF_L10.mat'))
    
    total_dev = []
    total_test = []
    
    for i in xrange(len(data_name)):
        if data_name[i] != train_data_name:
            X, y = preprocessing(loadmat(path + data_name[i] + '_SURF_L10.mat'))
            # split
            np.random.seed(seed)
            ind = np.random.permutation(range(X.shape[0]))
            N_train = int(X.shape[0] * ratio)
            total_dev.append((X[ind[:N_train]], y[ind[:N_train]]))
            total_test.append((X[ind[N_train:]], y[ind[N_train:]]))
            
    return X_train, y_train, total_dev, total_test
            
    
