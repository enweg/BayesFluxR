
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
the problems I am workin on. Extending things to classification problems
should be rather straight forward though, since BFlux allows custom
implementation of likelihoods and priors (at least in the Julia
version). In the R version presented here, this is currently not
possible.

## Basics

BFluxR’s and BFlux’s are based on Flux.jl and thus take over a lot of
their syntaxt. Before we demonstrate this, we first need to install and
load BFluxR. Installation is currently only possible from Github:

``` r
# devtools::install_github("enweg/BFlurR")
```

## Variational Inference

## MCMC Methods

## Current Problems and Shortcomings

## References
