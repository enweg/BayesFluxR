
#' Sample from the prior predictive of a Bayesian Neural Network
#'
#' @param bnn BNN obtained using \code{\link{BNN}}
#' @param n Number of samples
#'
#' @return matrix of prior predictive samples; Columns are the different samples
#' @examples
#' \dontrun{
#'   ## Needs previous call to `BayesFluxR_setup` which is time
#'   ## consuming and requires Julia and BayesFlux.jl
#'   BayesFluxR_setup(installJulia=TRUE, seed=123)
#'   net <- Chain(Dense(5, 1))
#'   like <- likelihood.feedforward_normal(net, Gamma(2.0, 0.5))
#'   prior <- prior.gaussian(net, 0.5)
#'   init <- initialise.allsame(Normal(0, 0.5), like, prior)
#'   x <- matrix(rnorm(5*100), nrow = 5)
#'   y <- rnorm(100)
#'   bnn <- BNN(x, y, like, prior, init)
#'   pp <- prior_predictive(bnn, n = 10)
#' }
#'
#' @export
prior_predictive <- function(bnn, n = 1){
  if (!JuliaCall::julia_exists("prior_predict_feedforward")){
    JuliaCall::julia_source(system.file("Julia/helpers-prior-predictive.jl", package = "BayesFluxR"))
  }

  if (ndims(bnn$x) == 2){
    values <- JuliaCall::julia_eval(sprintf("reduce(hcat, sample_prior_predictive(%s, prior_predict_feedforward(%s), %i))", bnn$juliavar, bnn$juliavar, n))
    return(values)
  }

  if (ndims(bnn$x) == 3){
    values <- JuliaCall::julia_eval(sprintf("reduce(hcat, sample_prior_predictive(%s, prior_predict_seqtoone(%s), %i))", bnn$juliavar, bnn$juliavar, n))
    return(values)
  }

  stop("Prior predictive not supported for your shape of x")
}


#' Draw from the posterior predictive distribution
#'
#' @param bnn a BNN obtained using \code{link{BNN}}
#' @param posterior_samples a vector or matrix containing posterior
#' samples. This can be obtained using \code{\link{mcmc}}, or \code{\link{bayes_by_backprop}}
#' or \code{\link{find_mode}}.
#' @param x input variables. If `NULL` (default), training values will be used.
#'
#' @return A matrix whose columns are the posterior predictive draws.
#' @examples
#' \dontrun{
#'   ## Needs previous call to `BayesFluxR_setup` which is time
#'   ## consuming and requires Julia and BayesFlux.jl
#'   BayesFluxR_setup(installJulia=TRUE, seed=123)
#'   net <- Chain(Dense(5, 1))
#'   like <- likelihood.feedforward_normal(net, Gamma(2.0, 0.5))
#'   prior <- prior.gaussian(net, 0.5)
#'   init <- initialise.allsame(Normal(0, 0.5), like, prior)
#'   x <- matrix(rnorm(5*100), nrow = 5)
#'   y <- rnorm(100)
#'   bnn <- BNN(x, y, like, prior, init)
#'   sampler <- sampler.SGLD()
#'   ch <- mcmc(bnn, 10, 1000, sampler)
#'   pp <- posterior_predictive(bnn, ch$samples)
#' }
#'
#' @export
posterior_predictive <- function(bnn, posterior_samples, x = NULL){
  if (is.null(x)){
    x <- bnn$x
  }
  sym.x <- get_random_symbol()
  JuliaCall::julia_assign(sym.x, x)
  JuliaCall::julia_command(sprintf("%s = Float32.(%s);",
                                   sym.x, sym.x))

  if (ndims(posterior_samples) == 3){
    stop("Tensors not supported. Call function for each slice.")
  }
  if (ndims(posterior_samples) == 1){
    posterior_samples <- matrix(posterior_samples, ncol = 1)
  }

  sym.theta <- get_random_symbol()
  JuliaCall::julia_assign(sym.theta, posterior_samples)
  JuliaCall::julia_command(sprintf("%s = Float32.(%s);",
                                   sym.theta, sym.theta))

  values <- JuliaCall::julia_eval(sprintf("sample_posterior_predict(%s, %s; x = %s)",
                                          bnn$juliavar, sym.theta, sym.x))

  return(values)
}
