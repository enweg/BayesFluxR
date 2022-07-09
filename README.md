
<!-- README.md is generated from README.Rmd. Please edit that file -->

# BFluxR

<!-- badges: start -->
<!-- badges: end -->

**Goals and Introduction**

BFluxR is a R interface to BFlux.jl which itself extends the famous
Flux.jl machine learning library in Julia. BFlux is meant to make
research and testing of Bayesian Neural Networks easy. It is not meant
for production, but rather for explorations. Currently only regression
problems are being kept in mind during the development, since those are
the problems I am working on. Extending things to classification
problems should be rather straight forward though, since BFlux allows
custom implementation of likelihoods and priors (at least in the Julia
version). In the R version presented here, this is currently not
possible.

## Basics

BFluxR and BFlux are based on Flux.jl and thus take over a lot of Flux’s
syntax. Before we demonstrate this, we first need to install and load
BFluxR. Installation is currently only possible from Github:

``` r
# devtools::install_github("enweg/BFluxR")
```

BFluxR depends on BFlux.jl which is a library written in Julia. Hence,
to run BFluxR, we need a way to access Julia. This is provided by the
JuliaCall library. So we also need to install that library. I recommend
installing a forked version, since the current official version installs
an old version of Julia, which is not supported by BFlux. This version
should be automatically installed when you install BFlux.

We are now ready to start exploring BFluxR. We first need to load the
package and run the setup. This will install Julia if you do not yet
have it and will install all the Julia dependencies, including BFlux.jl.
If you already do have Julia installed, then the script will pick the
Julia verion on your computer. If you, like me, have multiple versions
installed, then you can define the version you want to use by setting
the `JULIA_HOME` variable to the path of the Julia version you want to
use.

Running these lines for the first time can take a while, since it will
possibly have to install Julia and all dependencies. After this, if you
do no longer wish you check if all dependencies are available, you can
set `pkg_check = FALSE`.

We also set a working environment for Julia. This is generally good
practice and should in real projects be the project directory. It is
also good practice to already set a seed. This seed will be set in both
R and in Julia. If you wish to set a seed later on, please use
`.set_seed` which sets the seed in both Julia and R.

``` r
library(BFluxR)
#> 
#> Attaching package: 'BFluxR'
#> The following object is masked from 'package:stats':
#> 
#>     Gamma
BFluxR_setup(seed = 6150533, env_path = "/tmp", pkg_check = FALSE)
#> Julia version 1.7.3 at location /Users/enricowegner/Library/Application Support/org.R-project.R/R/JuliaCall/julia/1.7.3/Julia-1.7.app/Contents/Resources/julia/bin will be used.
#> Loading setup script for JuliaCall...
#> Finish loading setup script for JuliaCall.
#> Set the seed of Julia and R to 6150533
```

After loading BFluxR and running the setup, we are now ready to
experiment around. To create a Bayesian Neural Network, we first need a
Neural Network. In the Flux.jl context and thus also here, a network is
defined as a chain of layers. This is intuitively represented in the
syntax below, which creates a Feedforward Neural Network with one hidden
layer with `tanh` activation function. The last `Dense(1, 1)` statement
is the output connection. The chain below thus says: Feed a vector
![x](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;x "x")
into the network. Tranform this input via
![act=tanh(x'w_1 + b_1)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;act%3Dtanh%28x%27w_1%20%2B%20b_1%29 "act=tanh(x'w_1 + b_1)").
The output is then given by
![\\hat{y} = act'w_2 + b_2](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Chat%7By%7D%20%3D%20act%27w_2%20%2B%20b_2 "\hat{y} = act'w_2 + b_2").

``` r
net <- Chain(Dense(1, 1, "tanh"), Dense(1, 1))
```

To transform this standard Neural Network into a Baysian NN, we need a
likelihood and priors for all parameters. BFluxR view on priors and
likelihoods might be a bit unintuitive at the beginning: a prior
function only defines priors for the network parameters. Priors for all
parameters introduced by the likelihood, such as the standard deviation,
are defined in the likelihood functions. So, for example, we can use a
Gaussian prior for all network parameters

``` r
prior <- prior.gaussian(net, 0.5)  # sigma = 0.5
```

and can define a Gaussian likelihood. The Gaussian likelihood also
introduces a standard deviation though, which in BFlux terms must be
given a prior when defining the likelihood: it is not included in the
prior above, which applies only to the network parameters. Here we use a
Gamma prior for the standard deviation.

``` r
like <- likelihood.feedforward_normal(net, Gamma(2.0, 0.5))
```

For all of the estimation methods in BFluxR, we need some initial
values. BFluxR handles this via initialisers, which essentially just
tell it how to initialise all parameters. Currently only one kind of
initialiser is implemented: Initialising all parameters (network and
likelihood) by drawing randomly from a distribution. Users are free to
extend this in Julia, but this is currently not feasible in the R
version.

``` r
# Initialising all values by drawing from a Gaussian
init <- initialise.allsame(Normal(0, 0.5), like, prior)
```

Now that we have priors on all parameters as well as a likelihood, all
that is left is to have some data. Here we will take an AR(1) for the
simple reason that time series applications were the first application
for which BFluxR and BFlux were implemented.

``` r
n <- 600
y <- arima.sim(list(ar = 0.5), n = 600)
y.train <- y[1:floor(5/6*n)]
y.test <- y[(floor(5/6*n)+1):n]
x.test <- matrix(y.test[1:(length(y.test)-1)], nrow = 1) # In BFlux, rows are variables, columns are observations
x.train <- matrix(y.train[1:(length(y.train)-1)], nrow = 1)
y.test <- y.test[2:length(y.test)]
y.train <- y.train[2:length(y.train)]

plot.ts(y, xlab = "", ylab = "", main = "Simulated AR(1) Data")
abline(v = 500, lty = 2, lw = 2, col = "red")
text(x = 500, y = 2.5, labels = "End Training Data", col = "red")
```

<img src="man/figures/README-unnamed-chunk-8-1.png" width="100%" />

We now have everything to create a Bayesian Model. This is simply done
by calling `BNN` with the data, the likelihood, the prior, and the
initiliser. The prior and likelihood contain information about the
network structure, and thus it is not necessary to explicitly give the
network structure anymore.

``` r
bnn <- BNN(x.train, y.train, like, prior, init)
```

## Mode and Modal Approximations

The easiest way to test a BNN and to obtain a first estimate is by using
the MAP or the mode of the posterior distribution. BFluxR achieves this
by using stochastic gradient descent type of algorithms. Currently
implemented are `opt.ADAM`, `opt.RMSProp`, and `opt.Descent`.

``` r
# we are using ADAM with a batchsize of 10 and run for 1000 epochs
opt <- opt.ADAM()
mode <- find_mode(bnn, opt, 10, 1000)
```

We can use this mode estimat to draw from the posterior predictive.
Given that the mode is just a single value of the posterior, we will
replicate the mode estimate multiple times. This will still only
correspond to the one point in the posterior distribution, but will
deliver us multiple estimates (multiple draws from the posterior
characterised by the MAP).

``` r
post_draws <- matrix(rep(mode, 1000), ncol = 1000)
post_values.train <- posterior_predictive(bnn, post_draws, x = x.train)
post_values.test <- posterior_predictive(bnn, post_draws, x = x.test)

yhat.train <- apply(post_values.train, 1, mean)
yhat.test <- apply(post_values.test, 1, mean)
```

We can now check how good these predictions are:

``` r
mse.train <- mean((y.train - yhat.train)^2)
mse.test <- mean((y.test - yhat.test)^2)

c(mse.train, mse.test)
#> [1] 1.045913 1.049174
```

``` r
plot(y.test, type = "l", xlab = "Time", ylab = "", main = "Test data performance", col = 1)
lines(yhat.test, col = 2)
legend(x = 0, y = 3, c("y", "yhat"), col = c(1, 2), lty = c(1, 1))
```

<img src="man/figures/README-unnamed-chunk-13-1.png" width="100%" />

## Variational Inference

The next step up from a MAP estimate is a Variational Inference
estimate. BFluxR currently implements Bayes by Backprop (BBB)
(**welling?**) using a Multivariate Gaussian with diagonal covariance
matrix as proposal family. Although BBB is the standard in Bayesian NN,
it can at time be restrictive. For that reason, BFlux allows for
extensions in the Julia version, not currently though in the R version.

``` r
# using a batchsize of 10 and running for 1000 epochs
vi <- bayes_by_backprop(bnn, 10, 1000)
```

BBB does not itself return posterior draws yet. Instead, it returns
information about how the distribution looks like, restricted by it
being a Multivariate Gaussian with diagonal variance. To actually obtain
draws, we still need to sample from this variational family.

``` r
post_samples <- vi.get_samples(vi, n = 1000)
```

We can then use these draws like we would use any other posterior draws.
For BNNs we are usually not interested in the actual parameters, but
rather in the posterior predictive distribution. As such, we can use
these draws to obtain posterior predictive draws.

``` r
post_values.train <- posterior_predictive(bnn, post_samples, x = x.train)
post_values.test <- posterior_predictive(bnn, post_samples, x = x.test)

yhat.train <- apply(post_values.train, 1, mean)
upper.train <- apply(post_values.train, 1, function(x) quantile(x, 0.975))
lower.train <- apply(post_values.train, 1, function(x) quantile(x, 0.025))

yhat.test <- apply(post_values.test, 1, mean)
upper.test <- apply(post_values.test, 1, function(x) quantile(x, 0.975))
lower.test <- apply(post_values.test, 1, function(x) quantile(x, 0.025))


data.frame(
  mse.train = mean((y.train - yhat.train)^2),
  mse.test = mean((y.test - yhat.test)^2),
  coverage.train = mean(((y.train < upper.train) & (y.train > lower.train))),
  coverage.test = mean(((y.test < upper.test) & (y.test > lower.test)))
)
#>   mse.train mse.test coverage.train coverage.test
#> 1  1.051824 1.038127      0.9418838     0.9494949
```

## MCMC Methods

BFluxR implements various Markov Chain Monte Carlo methods that can be
use to obtain approximate draws from the posterior. Currently
implemented are

-   Stochastic Gradient Langevin Dynamics (SGLD): `sampler.SGLD`
-   Stochastic Gradient Nose-Hoover Thermostat (SGNHTS):
    `sampler.SGNHTS`
-   Gradient Guided Monte Carlo (GGMC): `sampler.GGMC`
-   Hamiltonian Monte Carlo (HMC): `sampler.HMC`
-   Adaptive Metropolis Hastings (AdaptiveMH): `sampler.AdaptiveMH`

Additionally, BFluxR allows for adaptation of mass matrices and
stepsizes for some of the samplers above:

-   Mass adaptation via `madapter.DiagCov`, `madapter.FullCov`,
    `madapter.FixedMassMatrix` or `madapter.RMSProp` can be done for
    SGNHTS, GGMC, and HMC
-   Step size adaptation via `sadapter.Const` or `sadapter.DualAverage`
    can be done for GGMC and HMC since these two are the only samplers
    using both gradients and a Metropolis-Hastings accept/reject step.

``` r
# sampling via sgld
sampler <- sampler.SGLD(stepsize_a = 1.0)
samples.sgld <- mcmc(bnn, 10, 10000, sampler) # batchsize 10 and 1000 samples

# sampling via SGNHTS
sampler <- sampler.SGNHTS(1e-2)
samples.sgnhts <- mcmc(bnn, 10, 10000, sampler)

# sampling using GGMC with fixed mass but stepsize adaptation
madapter = madapter.FixedMassMatrix()
sadapter = sadapter.DualAverage(1000, initial_stepsize = 1) # 1000 adaptation steps
sampler <- sampler.GGMC(l = 1, sadapter = sadapter, madapter = madapter)
samples.ggmc <- mcmc(bnn, 10, 10000, sampler)

# sampling using HMC
# This time batchsize is is length of y thus no batching
sadapter <- sadapter.DualAverage(1000, initial_stepsize = 0.001)
madapter <- madapter.FixedMassMatrix()
sampler <- sampler.HMC(l = 0.001, path_len = 5, sadapter = sadapter, madapter = madapter)
samples.hmc <- mcmc(bnn, length(y.train), 10000, sampler)

# sampling using Adaptive MH
# this does not use any gradients so we start at a mode 
sampler <- sampler.AdaptiveMH(bnn, 1000, 0.1)
samples.amh <- mcmc(bnn, 10, 10000, sampler, start_value = mode)
```

All of these samples can be used in the same way. For example, we can
use the samples obtained using SGNHTS to draw from the posterior
predictive. *We are again not really interested in the actual
parameters, but much more in the posterior predictive values and the
network output space*.

``` r
post_values.train <- posterior_predictive(bnn, samples.sgnhts$samples, x = x.train)
post_values.test <- posterior_predictive(bnn, samples.sgnhts$samples, x = x.test)


yhat.train <- apply(post_values.train, 1, mean)
upper.train <- apply(post_values.train, 1, function(x) quantile(x, 0.975))
lower.train <- apply(post_values.train, 1, function(x) quantile(x, 0.025))

yhat.test <- apply(post_values.test, 1, mean)
upper.test <- apply(post_values.test, 1, function(x) quantile(x, 0.975))
lower.test <- apply(post_values.test, 1, function(x) quantile(x, 0.025))


data.frame(
  mse.train = mean((y.train - yhat.train)^2),
  mse.test = mean((y.test - yhat.test)^2),
  coverage.train = mean(((y.train < upper.train) & (y.train > lower.train))),
  coverage.test = mean(((y.test < upper.test) & (y.test > lower.test)))
)
#>   mse.train mse.test coverage.train coverage.test
#> 1   1.04987 1.048984      0.9438878     0.9494949
```

It is often a good idea to check whether at least the parameters
additionally introduces by the likelihood have reasonable chains.
Although better statistics exist, here we will only check it visually.
Sigma is here given by the last parameter. In general, if the likelihood
introduces k parameters, then the last k parameters in each draw are
those of the likelihood.

``` r
s <- samples.sgnhts$samples
sigma <- s[nrow(s), ]
plot.ts(sigma)
```

<img src="man/figures/README-unnamed-chunk-19-1.png" width="100%" />

What if we were not happy or if the chain had not yet converged? In that
case we can just continue sampling instead of having to start fresh.

``` r
samples.sgnhts.cont <- mcmc(bnn, 10, 20000, sampler = samples.sgnhts$sampler, continue_sampling = TRUE)

list(
  dim(samples.sgnhts.cont$samples), 
  all.equal(samples.sgnhts$samples, samples.sgnhts.cont$samples[, 1:10000])
)
#> [[1]]
#> [1]     5 20000
#> 
#> [[2]]
#> [1] TRUE
```

## Recurrent Structures

## Overview: Estimation

BFluxR/BFlux currently implements Bayesian estimation on three different
accuracy levels. The first is modal approximations which are implemented
using Diagonal-multivariate-Gaussian Laplace approximations of a mode.
These can also be used as a mixture if the approximations are done for
multiple modes. This follows Gelman et al. (2013). Modal approximations
can be corrected using Sampling-Importance-Resampling, as suggested in
Gelman et al. (2013).

The second level of accuracy is given by Variational Approximations or
Variational Inference. Currently two VI methods are implemented:
Automatic Differentiation Variational Inference (ADVI) (Kucukelbir et
al. 2015; Ge, Xu, and Ghahramani 2018) and Bayes By Backprop (BBB)
(Blundell et al. 2015; Jospin et al. 2022). ADVI does currently not
allow for stochastic gradients and thus scales badly with the data set
size. It does allow for a much more flexible variational distribution
though. BBB, as currently implemented, does allow for stochastic
gradients and mini-batches and thus scales better, but currently only
supports diagonal-multivariate-Gaussians. Due to its scalability, BBB
can be estimated using various starting points though, allowing also
here to take a mixture of approximations. BFlux/BFluxR allows to do this
in parallel.

The third level of approximations is given by Markov Chain Monte Carlo
Methods (MCMC). Two methods are currently implemented. The most natural
to most machine learning researchers due to its similarity to standard
Neural Network estimation is Stochastic Gradient Langevin Dynamics as
introduced by Welling and Teh (2011). SGLD could be summarised as doing
SGD with additional gradient noise in each iteration. Welling and Teh
(2011) show that as the step size goes to zero, the discrete SGLD
algorithm converges to Langevin Dynamics which have the posterior as
their stationary distribution. Garriga-Alonso and Fortuin (2021)
recently showed though, that although the statement is true, for any
stepsize greater than zero (and thus essentially always), SGLD and other
SGMCMC (Nemeth and Fearnhead 2021) methods have a zero
Metropolis-Hastings (MH) acceptance probability. They point out that
this is due to the Euler-Maruyama method used to discretesize Langevin
Dynamics. Due to the zero MH acceptance probability, SGLD cannot be
corrected for the discretization error using the common MH scheme and is
thus difficult to monitor. So SGLD should always be just seen as another
approximation with (in practical scenarios) essentially no convergence
guarantees. As a remedy to this shortcoming Garriga-Alonso and Fortuin
(2021) propose the use of Gradient Guided Monte Carlo (GGMC) which can
be, and is in BFluxR/BFlux, implemented using stochastic gradient
methods. GGMC uses a reversible integrator and thus the MH acceptance
probability is positive. Moreoever, the mass matrix and step length can
be tuned along the the lines of standard Hamiltonian Monte Carlo
adaptations/tuning. BFluxR/BFlux currently implements these tunings,
although in a simplified way currently.

## Current Problems and Shortcomings

## References

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-bayesbybackprop" class="csl-entry">

Blundell, Charles, Julien Cornebise, Koray Kavukcuoglu, and Daan
Wierstra. 2015. “Weight Uncertainty in Neural Network.” In
*International Conference on Machine Learning*, 1613–22. PMLR.

</div>

<div id="ref-garriga2021exact" class="csl-entry">

Garriga-Alonso, Adrià, and Vincent Fortuin. 2021. “Exact Langevin
Dynamics with Stochastic Gradients.” *arXiv Preprint arXiv:2102.01691*.

</div>

<div id="ref-turing" class="csl-entry">

Ge, Hong, Kai Xu, and Zoubin Ghahramani. 2018. “Turing: A Language for
Flexible Probabilistic Inference.” In *International Conference on
Artificial Intelligence and Statistics, AISTATS 2018, 9-11 April 2018,
Playa Blanca, Lanzarote, Canary Islands, Spain*, 1682–90.
<http://proceedings.mlr.press/v84/ge18b.html>.

</div>

<div id="ref-bda3" class="csl-entry">

Gelman, Andrew, John B. Carlin, Hal S. Stern, David B. Dunson, Aki
Vehtari, and Donald B. Rubin. 2013. *Bayesian Data Analysis*. 3rd ed.
Chapman &amp; Hall/CRC.

</div>

<div id="ref-jospin2022hands" class="csl-entry">

Jospin, Laurent Valentin, Hamid Laga, Farid Boussaid, Wray Buntine, and
Mohammed Bennamoun. 2022. “Hands-on Bayesian Neural Networks—a Tutorial
for Deep Learning Users.” *IEEE Computational Intelligence Magazine* 17
(2): 29–48.

</div>

<div id="ref-advi2015" class="csl-entry">

Kucukelbir, Alp, Rajesh Ranganath, Andrew Gelman, and David Blei. 2015.
“Automatic Variational Inference in Stan.” *Advances in Neural
Information Processing Systems* 28.

</div>

<div id="ref-nemeth2021stochastic" class="csl-entry">

Nemeth, Christopher, and Paul Fearnhead. 2021. “Stochastic Gradient
Markov Chain Monte Carlo.” *Journal of the American Statistical
Association* 116 (533): 433–50.

</div>

<div id="ref-welling2011bayesian" class="csl-entry">

Welling, Max, and Yee W Teh. 2011. “Bayesian Learning via Stochastic
Gradient Langevin Dynamics.” In *Proceedings of the 28th International
Conference on Machine Learning (ICML-11)*, 681–88. Citeseer.

</div>

</div>
