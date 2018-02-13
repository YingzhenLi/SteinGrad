import os
import tempfile
import urllib
import numpy as np
from scipy.misc import imsave
import cPickle
import math
import import_data_mnist
from scipy.io import loadmat

def mnist(datasets_dir='/TMP/'):
    URL_MAP = {
    "train": "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat",
    "valid": "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat",
    "test": "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat"
    }

    PATH_MAP = {
    "train": os.path.join(tempfile.gettempdir(), "binarized_mnist_train.npy"),
    "valid": os.path.join(tempfile.gettempdir(), "binarized_mnist_valid.npy"),
    "test": os.path.join(tempfile.gettempdir(), "binarized_mnist_test.npy")
    }
    for name, url in URL_MAP.items():
        local_path = PATH_MAP[name]
        if not os.path.exists(local_path):
            np.save(local_path, np.loadtxt(urllib.urlretrieve(url)[0]))

    train_set = [x for x in np.load(PATH_MAP['train'])]
    valid_set = [x for x in np.load(PATH_MAP['valid'])]
    test_set =  [x for x in np.load(PATH_MAP['test'])]

    x_train = np.array(train_set).astype(np.float32)
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_valid = np.array(valid_set).astype(np.float32)
    x_valid = x_valid.reshape(x_valid.shape[0], 1, 28, 28)
    x_test = np.array(test_set).astype(np.float32)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
    
    # for tensorflow
    x_train = np.transpose(x_train, (0, 2, 3, 1))
    x_valid = np.transpose(x_valid, (0, 2, 3, 1))
    x_test = np.transpose(x_test, (0, 2, 3, 1))

    return x_train, x_valid, x_test

def mnist_aug(path, ratio = 0.9, seed = 0, digits = None):
    # load and split data
    print "Loading data"
    path = path + 'MNIST/'
    data_train, labels_train = import_data_mnist.read(path, 0, "training", seed, digits)
    data_test, labels_test = import_data_mnist.read(path, 0, "testing", seed, digits)
    #data_train = np.array(data >= 0.5 * np.max(data, 0), dtype = int)	# binary
    #data_test = np.array(data >= 0.5 * np.max(data, 0), dtype = int)	# binary
    data_train /= 255.0	# real-value
    data_test /= 255.0	# real-value
    # transform to float32
    data_train = np.array(data_train.T, dtype='f')	# float32
    data_test = np.array(data_test.T, dtype='f')	# float32
    labels_train = np.array(labels_train.T, dtype='f')	# float32
    labels_test = np.array(labels_test.T, dtype='f')	# float32

    shape=(28, 28)
    data_train = data_train.reshape((data_train.shape[0],) + shape + (1,))
    data_test = data_test.reshape((data_test.shape[0],) + shape + (1,))

    return data_train, data_test, labels_train, labels_test

def omniglot(path, ratio = 0.9, seed = 0):
    # load and split data
    print "Loading data"
    mat = loadmat(path + 'OMNIGLOT/chardata.mat')
    data_train = np.array(mat['data'].T, dtype='f')     # float32
    data_test = np.array(mat['testdata'].T, dtype='f')  # float32
    labels_train = np.array(mat['target'].T, dtype='f') # float32
    labels_test = np.array(mat['testtarget'].T, dtype='f')      # float32
    
    shape=(28, 28)
    data_train = data_train.reshape((data_train.shape[0],) + shape + (1,))
    data_test = data_test.reshape((data_test.shape[0],) + shape + (1,))

    return data_train, data_test, labels_train, labels_test

def cifar10(path, return_label = False):
    # load and split data
    def unpickle(path, name):
        f = open(path + 'cifar-10-batches-py/' + name,'rb')
        data = cPickle.load(f)
        f.close()
        return data
    def futz(X):
        return X.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
        
    print "Loading data"
    data_train = np.zeros((50000, 32, 32, 3), dtype='uint8')
    labels_train = np.zeros(50000, dtype='int32')
    fnames = ['data_batch_%i'%i for i in range(1,6)]

    # load train and validation data
    n_loaded = 0
    for i, fname in enumerate(fnames):
        data = unpickle(path, fname)
        assert data['data'].dtype == np.uint8
        data_train[n_loaded:n_loaded + 10000] = futz(data['data'])
        labels_train[n_loaded:n_loaded + 10000] = data['labels']
        n_loaded += 10000
    
    # load test set
    data = unpickle(path, 'test_batch')
    assert data['data'].dtype == np.uint8
    data_test = futz(data['data'])
    labels_test = data['labels']
    
    # convert to float
    data_train = np.array(data_train, dtype='f')	# float32
    data_test = np.array(data_test, dtype='f')	# float32
    labels_train = np.array(labels_train, dtype='f')	# float32
    labels_test = np.array(labels_test, dtype='f')
    
    data_train = 1.0 * data_train / 256.
    data_test = 1.0 * data_test / 256.
    
    if return_label:
        return data_train, data_test, labels_train, labels_test
    else:
        return data_train, data_test
    
    
def grayscale_grid_vis(X, (nh, nw), save_path=None):
    h, w = X[0].shape[:2]
    img = np.zeros((h*nh, w*nw))
    for n, x in enumerate(X):
        j = n/nw
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w] = x
    if save_path is not None:
        imsave(save_path, img)
    return img

def color_plot_images(images, shape, path, filename, n_rows = 10, color = True):
     # finally save to file
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    images = reshape_and_tile_images(images, shape, n_rows)
    if color:
        from matplotlib import cm
        plt.imsave(fname=path+filename+".png", arr=images, cmap=cm.Greys_r)
    else:
        plt.imsave(fname=path+filename+".png", arr=images, cmap='Greys')
    #plt.axis('off')
    #plt.tight_layout()
    #plt.savefig(path + filename + ".png", format="png")
    print "saving image to " + path + filename + ".png"
    plt.close()

def reshape_and_tile_images(array, shape=(28, 28), n_rows=None):
    if n_rows is None:
        n_rows = int(math.sqrt(array.shape[0]))
    n_cols = int(math.ceil(float(array.shape[0])/n_rows))
    if len(shape) == 2:
        order = 'C'
    else:
        order = 'F'
    
    def cell(i, j):
        ind = i*n_cols+j
        if i*n_cols+j < array.shape[0]:
            return array[ind].reshape(shape, order=order)
        else:
            return np.zeros(shape)

    def row(i):
        return np.concatenate([cell(i, j) for j in range(n_cols)], axis=1)

    return np.concatenate([row(i) for i in range(n_rows)], axis=0)

def iter_data(*data, **kwargs):
    size = kwargs.get('size', 128)
    try:
        n = len(data[0])
    except:
        n = data[0].shape[0]
    batches = n / size
    #if n % size != 0:
    #    batches += 1

    for b in range(batches):
        start = b * size
        end = (b + 1) * size
        if end > n:
            end = n
        if len(data) == 1:
            yield data[0][start:end]
        else:
            yield tuple([d[start:end] for d in data])
    
