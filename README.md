# Stein Gradient Estimator

Thank you for your interest in our paper:

[Yingzhen Li](http://yingzhenli.net) and 
[Richard E. Turner](http://cbl.eng.cam.ac.uk/Public/Turner/WebHome)

[Gradient Estimators for Implicit Models](https://openreview.net/forum?id=SJi9WOeRb)

International Conference on Learning Representations (ICLR), 2018

Roughly speaking, whenever you need to compute dlogp(x)/dx you can use our method. Applications include: variational inference, maximum entropy, gradient-based MCMC, entropy regularisation (to GANs), and more...

Please consider citing the paper when any of the material is used for your research.

Contributions: Yingzhen derived the estimator and implemented all experiments. Rich provided advices for paper writing. Other people who provided comments on the manuscript are acknowledged in the paper.

## Experiments

I've got three experiments to demonstrate the wide application of the gradient estimator. In the folders you can find the corresponding code, each accompanied by another README.md for further details.

Also the kernel induced hamiltonian flow code depends on another package, see [https://github.com/karlnapf/kernel_hmc](https://github.com/karlnapf/kernel_hmc)

## Citing the paper (bib)
```
@inproceedings{li2018gradient,
  title = {Gradient Estimators for Implicit Models},
  author = {Li, Yingzhen and Turner, Richard E.},
  booktitle = {International Conference on Learning Representations},
  year = {2018}
}
```
