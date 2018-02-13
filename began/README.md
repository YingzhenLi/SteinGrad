# Entropy regularisation for BEGAN

This repo provides code for the entropy regularised GAN experiments. The idea is very simple, for any generator p(x) and any 
GAN generator loss L(p), we can use entropy regularisation to further encourage diversity:

Loss(p) = L(p) - alpha * H(p)

As argued in the paper we only need gradients of Loss(p), which means we don't need to estimate H(p), and instead we just need
estimation of dlogp(x)/dx.

The GAN algorithm we tested is the Boundary Equilibrium GAN (BEGAN, [Berthelot et al. 2017](https://arxiv.org/abs/1703.10717)).
It is slight tricky here because this algorithm does require estimating H(p) to maintain equilibrium. So we introduced two
strategies to do so, but still used the gradient estimators to compute the backprop gradient.

To run the code, first you need to download MNIST [here](http://yann.lecun.com/exdb/mnist/). 
You also need python packages [tqdm](https://pypi.python.org/pypi/tqdm) 
and [shutil](https://docs.python.org/2/library/shutil.html).

When you have data and packages ready, you can modify lines 38 and 124 of 
[mnist_exp.py](https://github.com/YingzhenLi/SteinGrad/blob/master/began/mnist_exp.py), and run

    python mnist_exp.py method entropy_option gamma alpha
    
where method can be one of the following:

'original': run the original BEGAN, here entropy_option and alpha is ineffective

'kde': run entropy regularised BEGAN with KDE plugin gradient estimator

'score': run entropy regularised BEGAN with score matching gradient estimator

'stein': run entropy regularised BEGAN with Stein gradient estimator

For methods other than 'orginal', the entropy_option can be one of the following:

'kdeH': using KDE for entropy estimate

'proxyH': a proxy loss using first-order Taylor expansion + MC estimate, see details in the paper

Lastly gamma is a hyper-parameter for BEGAN and alpha is the hyper-parameter to control entropy regularisation.
