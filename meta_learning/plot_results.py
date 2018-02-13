import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True

def read_results(dataset, options):
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
        results[option] = [nll, err, ksd]
        print nll.mean(), err.mean(), ksd.mean()
        
    return results

if __name__ == "__main__":
    datasets = ['australian', 'breast', 'pima']#, 'sonar']#, 'power', 'kin8nm', 'naval']
    #options = ['kde', 'score', 'stein', 'sgld', 'map']
    options = ['score_hsquare1.00_lbd0.5', 'score_hsquare10.00_lbd1.0', 'score_hsquare-1.00_lbd2.0']#, 'score_hsquare4.00_lbd0.1']#, 'score_hsquare10.0']
    results = {}
    for dataset in datasets:
        results[dataset] = read_results(dataset, options)
        
    # plot results
    color = {'score_hsquare1.00_lbd0.5': '#89fe05', 'score_hsquare-1.00_lbd2.0': '#01ff07', 
             'score_hsquare10.00_lbd1.0':'#04d8b2', 'score_hsquare10.0':'#3f9b0b', 'map': 'c'}
    label = {'score_hsquare1.00_lbd0.5': r'$\sigma^2 = 1.0, \eta = 0.5$', 
             'score_hsquare-1.00_lbd2.0': r'$median\ trick, \eta = 2.0$', 
             'score_hsquare10.00_lbd1.0': r'$\sigma^2 = 10.0, \eta = 1.0$', 
             'score_hsquare5.0': r'$\sigma^2 = 5.0$', 'score_hsquare10.0': r'$\sigma^2 = 10.0$',
             'sgld': r'$SGLD$', 'map': r'$MAP$'}
    f, ax = plt.subplots(3, len(datasets), figsize = (10, 5))
    i = 0
    x_axis = np.arange(0, 200, 10)
    for dataset in datasets:
        result = results[dataset]
        for option in options:
            time = result[option][2]
            
            # test nll
            nll_mean = result[option][0].mean(0)
            nll_ste = np.sqrt(result[option][0].var(0) / 5)
            x_axis = np.arange(1, nll_mean.shape[0]+1) * 50
            if dataset == 'sonar': x_axis *= 2
            ax[0, i].plot(x_axis, nll_mean, color=color[option], linewidth=2, \
                          label = label[option])
            #ax[0, i].fill_between(x_axis, nll_mean - nll_ste, nll_mean + nll_ste, \
            #                      color=color[option], alpha=0.2, linewidth=0) 
            ax[0, i].set_title(r'$%s$'%dataset)
            ax[0, 0].set_ylabel(r'$neg. \ LL$')
            # error  
            err_mean = result[option][1].mean(0)
            err_ste = np.sqrt(result[option][1].var(0) / 5)
            ax[1, i].plot(x_axis, err_mean, color=color[option], linewidth=2, \
                          label = label[option])
            #ax[1, i].fill_between(x_axis, err_mean - err_ste, err_mean + err_ste, \
            #                      color=color[option], alpha=0.2, linewidth=0)
            ax[1, 0].set_ylabel(r'$test \ error$')
            ax[0, 1].legend(loc='upper left', bbox_to_anchor=(-1.1,1.45, 0., 0.), ncol=3, \
                            borderaxespad=0., frameon=False)
            # ksd  
            ksd_mean = result[option][2].mean(0)
            ksd_ste = np.sqrt(result[option][2].var(0) / 5)
            ax[2, i].plot(x_axis, ksd_mean, color=color[option], linewidth=2)
            #ax[2, i].fill_between(x_axis, ksd_mean - ksd_ste, ksd_mean + ksd_ste, \
            #                      color=color[option], alpha=0.2, linewidth=0)
            ax[2, i].set_xlabel(r'$iteration$')
            ax[2, 0].set_ylabel(r'$KSD / dim(\theta)$')
        i+=1
    #plt.show()
    plt.savefig('results_compare.pdf', format='pdf')

