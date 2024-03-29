% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/utils.R
\name{to_bayesplot}
\alias{to_bayesplot}
\title{Convert draws array to conform with `bayesplot`}
\usage{
to_bayesplot(ch, param_names = NULL)
}
\arguments{
\item{ch}{Chain of draws obtained using \code{\link{mcmc}}}

\item{param_names}{If `NULL`, the parameter names will be of the
form `param_1`, `param_2`, etc. If `param_names` is a string,
the parameter names will start with the string with the number
of the parameter attached to it. If `param_names` is a vector, it
has to provide a name for each paramter in the chain.}
}
\value{
Returns an array of dimensions draws x chains x params.
}
\description{
BayesFluxR returns draws in a matrix of dimension
params x draws. This cannot be used with the `bayesplot` package
which expects an array of dimensions draws x  chains x params.
}
\examples{
\dontrun{
  ## Needs previous call to `BayesFluxR_setup` which is time
  ## consuming and requires Julia and BayesFlux.jl
  BayesFluxR_setup(installJulia=TRUE, seed=123)
  net <- Chain(Dense(5, 1))
  like <- likelihood.feedforward_normal(net, Gamma(2.0, 0.5))
  prior <- prior.gaussian(net, 0.5)
  init <- initialise.allsame(Normal(0, 0.5), like, prior)
  x <- matrix(rnorm(5*100), nrow = 5)
  y <- rnorm(100)
  bnn <- BNN(x, y, like, prior, init)
  sampler <- sampler.SGLD()
  ch <- mcmc(bnn, 10, 1000, sampler)
  ch <- to_bayesplot(ch)
  library(bayesplot)
  mcmc_intervals(ch, pars = paste0("param_", 1:10))
}
}
