import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True

def read_results(dataset, options):
    #path = 'iclr_save/results/'
    path = 'results/'
    
    results = {}
    for option in options:
        print 'result from', dataset, option
        filename = "bnn_%s_%s.pkl" % (dataset, option)
        a = pickle.load(open(path + filename, 'r'))
        print 'open', path+filename
        nll = -np.asarray(a['ll'])
        err = 1.0-np.asarray(a['acc'])
        ksd = np.asarray(a['ksd'])
        #if option == 'sgld' and dataset != 'sonar':
        #    nll = nll[:, :20]; err = err[:, :20]; ksd = ksd[:, :20]
        #    print 'sgld', nll.shape, err.shape, ksd.shape
        results[option] = [nll, err, ksd]
        print nll.mean(), err.mean(), ksd.mean()
        
    return results

if __name__ == "__main__":
    datasets = ['australian', 'breast', 'pima', 'sonar']#, 'power', 'kin8nm', 'naval']
    #options = ['kde', 'score', 'stein', 'sgld', 'map']
    options = ['kde_hsquare-1.00_lbd0.01', 'score_hsquare1.00_lbd1.00', 'stein_hsquare-1.00_lbd0.01', 
               'map', 'sgld']

    #options = ['sgld', 'amc']

    results = {}
    for dataset in datasets:
        results[dataset] = read_results(dataset, options)
        
    # plot results
    color = ['r', 'g', 'b', 'c', 'm']
    label = [r'$KDE$', r'$Score$', r'$Stein$', r'$MAP$', r'$SGLD$']
    f, ax = plt.subplots(3, len(datasets), figsize = (10, 5))
    i = 0
    x_axis = np.arange(0, 200, 10)
    for dataset in datasets:
        result = results[dataset]
        j = 0
        for option in options:
            time = result[option][2]
            # test nll
            nll_mean = result[option][0].mean(0)
            nll_ste = np.sqrt(result[option][0].var(0) / 5)
            #if option != 'sgld':
            x_axis = np.arange(1, nll_mean.shape[0]+1) * 50
            #else:
            #    x_axis = np.arange(1, nll_mean.shape[0]+1) * 100
            if dataset == 'sonar': x_axis *= 2
            ax[0, i].plot(x_axis, nll_mean, color=color[j], linewidth=2)
            ax[0, i].fill_between(x_axis, nll_mean - nll_ste, nll_mean + nll_ste, \
                                  color=color[j], alpha=0.2, linewidth=0) 
            ax[0, i].set_title(r'$%s$'%dataset)
            ax[0, 0].set_ylabel(r'$neg. \ LL$')
            # error  
            err_mean = result[option][1].mean(0)
            err_ste = np.sqrt(result[option][1].var(0) / 5)
            ax[1, i].plot(x_axis, err_mean, color=color[j], linewidth=2)
            ax[1, i].fill_between(x_axis, err_mean - err_ste, err_mean + err_ste, \
                                  color=color[j], alpha=0.2, linewidth=0)
            ax[1, 0].set_ylabel(r'$test \ error$')
            # ksd  
            ksd_mean = result[option][2].mean(0)
            ksd_ste = np.sqrt(result[option][2].var(0) / 5)
            ax[2, i].plot(x_axis, ksd_mean, color=color[j], label=label[j], linewidth=2)
            ax[2, i].fill_between(x_axis, ksd_mean - ksd_ste, ksd_mean + ksd_ste, \
                                  color=color[j], alpha=0.2, linewidth=0)
            ax[2, i].set_xlabel(r'$iteration$')
            ax[2, 0].set_ylabel(r'$KSD / dim(\theta)$')

            j += 1

        i += 1
        j = 0
    ax[2, 1].legend(loc='upper right', ncol=1, \
                            borderaxespad=0., frameon=False)
    ax[0, 3].set_xticks([0, 2500, 5000])
    ax[1, 3].set_xticks([0, 2500, 5000])
    ax[2, 3].set_xticks([0, 2500, 5000])

    plt.tight_layout()

    #plt.show()
    plt.savefig('results_methods.pdf', format='pdf')

