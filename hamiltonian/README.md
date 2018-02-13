# Kernel Induced Hamiltonian Dynamics

Tested with (kernel based) score matching and Stein gradient estimator.

This repo provides a demo for gradient-free HMC using gradient estimators. 

In short, HMC requires the gradient dlogp(x)/dx of the target density p(x), which might not be available always (e.g. in ABC).
The kernel induced hamiltonian dynamcis ([Strathman et al. 2015](https://arxiv.org/abs/1506.02564)) simply replace the gradient 
in HMC with a kernel based gradient estimator, so this allows us to run approximate posterior sampling.

My implementation here are largely based on the [original code](https://github.com/karlnapf/kernel_hmc), thanks to their 
awesome work!

To run the test, run

    python demo_trajectories.py seed
    python plot_results.py seed

where seed is an integer representing a random seed. You can also play around with the hyper-parameters inside
[demo_trajectories.py](https://github.com/YingzhenLi/SteinGrad/blob/master/hamiltonian/demo_trajectories.py).
