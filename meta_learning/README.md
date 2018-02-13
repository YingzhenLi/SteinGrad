# Meta-learning for approximate posterior samplers

Why not try to think about how meta-learning can help learning inference methods, such as approximate posterior samplers? :)

As far as I know, this is the first attempt that tried meta-learning for (approximate) posterior samplers 
(first submission done in May 2017). There are two very recent papers -- [Song et al. 2017](https://docs.python.org/2/library/shutil.html) 
and [Levy et al. 2017](https://arxiv.org/abs/1711.09268) -- 
that also investigate this task, but the ways they do it are very different.

Essentially the inspiration comes from SGLD ([Welling and Teh 2011](https://dl.acm.org/citation.cfm?id=3104568)) and 
Learning by gradient descent "square" ([Andrychowicz et al. 2016](https://arxiv.org/abs/1606.04474)). We know that SG-MCMC in
some sense can be viewed as "gradient descent + properly scaled noise", 
also ([Andrychowicz et al. 2016](https://arxiv.org/abs/1606.04474) proposed a meta-learning method to learn the gradient descent
part. So I thought it would be interesting to try learning an SG-MCMC-like approximate sampler using the approach of 
([Andrychowicz et al. 2016](https://arxiv.org/abs/1606.04474), but it also requires replacing the objective to variational 
lower-bound.

Then the variational lower-bound is intractable due to the implicitly defined distribution of the approximate sampler. 
So it's a natural idea to try our gradient estimator here!

To see how it works, run

    python exp_nn.py method train_data test_data seed hsquare eta
    
and here we support 4 methods:

'map': use MAP objective and ignore the entropy term, none of the gradient estimator will be used

'kde': use variational lower-bound and KDE plugin estimator for the entropy gradient

'score': use variational lower-bound and score matching gradient estimator for the entropy gradient

'stein': use variational lower-bound and Stein gradient estimator for the entropy gradient

train_data and test_data are the names of the datasets that used to train/test the sampler.

seed is the integer for the random seed

hsquare and eta are hyper-parameters of the (kernel-based) gradient estimators, see paper for details.

You can also run 

    python exp_sgld.py test_data seed
    
to see the performance of SGLD.
