
#' Use an isotropic Gaussian prior
#'
#' Use a Multivariate Gaussian prior for all network parameters.
#' Covariance matrix is set to be equal `sigma * I` with `I` being
#' the identity matrix. Mean is zero.
#'
#' @param chain Chain obtained using \code{\link{Chain}}
#' @param sigma Standard deviation of Gaussian prior
#'
#' @return a list containing the following
#' \itemize{
#'     \item `juliavar` the julia variable used to store the prior
#'     \item `juliacode` the julia code
#' }
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
#' }
#'
#' @export
prior.gaussian <- function(chain, sigma){
  juliacode <- sprintf("GaussianPrior(%s, Float32(%e))",
                       chain$nc, sigma)
  juliavar <- get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = %s;",
                                   juliavar, juliacode))
  out <- list(juliavar = juliavar, juliacode = juliacode)
  return(out)
}

#' Scale Mixture of Gaussian Prior
#'
#' Uses a scale mixture of Gaussian for each network parameter. That is,
#' the prior is given by
#' \deqn{\pi_1 Normal(0, sigma1) + (1-\pi_1) Normal(0, sigma2)}
#'
#' @param chain Chain obtained using \code{\link{Chain}}
#' @param sigma1 Standard deviation of first Gaussian
#' @param sigma2 Standard deviation of second Gaussian
#' @param pi1 Weight of first Gaussian
#'
#' @return a list containing the following
#' \itemize{
#'     \item `juliavar` the julia variable used to store the prior
#'     \item `juliacode` the julia code
#' }
#' @examples
#' \dontrun{
#'   ## Needs previous call to `BayesFluxR_setup` which is time
#'   ## consuming and requires Julia and BayesFlux.jl
#'   BayesFluxR_setup(installJulia=TRUE, seed=123)
#'   net <- Chain(Dense(5, 1))
#'   like <- likelihood.feedforward_normal(net, Gamma(2.0, 0.5))
#'   prior <- prior.mixturescale(net, 10, 0.1, 0.5)
#'   init <- initialise.allsame(Normal(0, 0.5), like, prior)
#'   x <- matrix(rnorm(5*100), nrow = 5)
#'   y <- rnorm(100)
#'   bnn <- BNN(x, y, like, prior, init)
#'   sampler <- sampler.SGLD()
#'   ch <- mcmc(bnn, 10, 1000, sampler)
#' }
#'
#' @export
prior.mixturescale <- function(chain, sigma1, sigma2, pi1){
  juliacode <- sprintf("MixtureScalePrior(%s, Float32(%e), Float32(%e), Float32(%e))",
                       chain$nc, sigma1, sigma2, pi1)
  juliavar <- get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = %s;",
                                   juliavar, juliacode))
  out <- list(juliavar = juliavar, juliacode = juliacode)
  return(out)
}
