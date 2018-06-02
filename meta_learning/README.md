# Meta-learning for approximate posterior samplers

Why not try to think about how meta-learning can help learning inference methods, 
such as training posterior samplers that can generalise to unseen tasks? :)

## improved version (update: Jun 2018)

Here I include a new **SG-MCMC sampler** that is learnable via meta-learning. 
You can have a look at the two files in [sampler/](sampler/) and see how they differ from each other. 
The meta-leanred **SG-MCMC sampler** is generally better than the meta-learned **approximate sampler**.

[Wenbo Gong](http://mlg.eng.cam.ac.uk/?portfolio=wenbo-gong) and I recently developed an even more advanced 
version of meta-learnable **SG-MCMC sampler** than the version included here. 
With that latest version we can finally do BNN classification on MNIST and also Bayesian RNNs!
We will release the paper and pyTorch code very soon :) There will be a link here pointing to that repo.

For other updates, I improved the orginal code a bit more to make it cleaner. 
Also I fixed a few bugs and added a new technique that is a bit like "experience reply" in deep RL, 
now the results for the **approximate samplers** are even better than reported in the ICLR camera ready.

To test the current version:

Training:

    python train_grad_uci.py --method stein --task crabs

This command will train an approximate sampler on the crabs data, using the Stein gradient estimator to approximate the entropy gradient.
See the code for the usage of more arguments.

Once trained a sampler, to test it, run

    python test_uci.py --method stein --task sonar


=====================================================

As far as I know, this is the first attempt that tried meta-learning for (approximate) posterior samplers 
(first arXiv submission in May 2017). There are two very recent papers -- [Song et al. 2017](https://docs.python.org/2/library/shutil.html) 
and [Levy et al. 2017](https://arxiv.org/abs/1711.09268) -- 
that also investigate this task, but the ways they do it are very different. 
Their experimental results are very good, but their approaches do not generalise to different dimensions. We do.

Essentially the inspiration comes from SGLD ([Welling and Teh 2011](https://dl.acm.org/citation.cfm?id=3104568)) and 
Learning by gradient descent "square" ([Andrychowicz et al. 2016](https://arxiv.org/abs/1606.04474)). We know that SG-MCMC in
some sense can be viewed as "gradient descent + properly scaled noise", 
also ([Andrychowicz et al. 2016](https://arxiv.org/abs/1606.04474) proposed a meta-learning method to learn the gradient descent
part. So I thought it would be interesting to try learning an SG-MCMC-like approximate sampler using the approach of 
([Andrychowicz et al. 2016](https://arxiv.org/abs/1606.04474), but it also requires replacing the objective to variational 
lower-bound.

Then the variational lower-bound is intractable due to the implicitly defined distribution of the approximate sampler. 
So it's a natural idea to try our gradient estimator here!

