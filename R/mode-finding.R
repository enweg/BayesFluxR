# If you want to implement more, check out the documentation for
# Flux at https://fluxml.ai/Flux.jl/stable/training/optimisers/
# and follow the examples below.

#' Standard gradient descent
#'
#' @param eta stepsize
#'
#' @return list containing
#' \itemize{
#'     \item `julivar` - julia variable holding the optimiser
#'     \item `juliacode` - string representation
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
#'   find_mode(bnn, opt.Descent(1e-5), 10, 100)
#' }
#'
#' @export
opt.Descent <- function(eta = 0.1){
  juliavar <- get_random_symbol()
  juliacode <- sprintf("Flux.Descent(%e)",
                       eta)

  JuliaCall::julia_command(sprintf("%s = %s;",
                                   juliavar, juliacode))
  out <- list(juliavar = juliavar, juliacode = juliacode)
  return(out)
}

#' ADAM optimiser
#'
#' @param eta stepsize
#' @param beta momentum decays; must be a list of length 2
#' @param eps Flux does not document this
#'
#' @return see \code{\link{opt.Descent}}
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
#'   find_mode(bnn, opt.ADAM(), 10, 100)
#' }
#'
#' @export
opt.ADAM <- function(eta = 0.001, beta = c(0.9, 0.999), eps = 1e-8){
  juliavar <- get_random_symbol()
  juliacode <- sprintf("Flux.ADAM(%e, (%e, %e), %e)",
                       eta, beta[1], beta[2], eps)
  JuliaCall::julia_command(sprintf("%s = %s;",
                                   juliavar, juliacode))
  out <- list(juliavar = juliavar, juliacode = juliacode)
  return(out)
}

#' RMSProp optimiser
#'
#' @param eta learning rate
#' @param rho momentum
#' @param eps not documented by Flux
#'
#' @return see \code{\link{opt.Descent}}
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
#'   find_mode(bnn, opt.RMSProp(), 10, 100)
#' }
#'
#' @export
opt.RMSProp <- function(eta = 0.001, rho = 0.9, eps = 1e-8){
  juliavar <- get_random_symbol()
  juliacode <- sprintf("Flux.RMSProp(%e, %e, %e)",
                       eta, rho, eps)
  JuliaCall::julia_command(sprintf("%s = %s;",
                                   juliavar, juliacode))
  out <- list(juliavar = juliavar, juliacode = juliacode)
  return(out)
}

#' Find the MAP of a BNN using SGD
#'
#' @param bnn a BNN obtained using \code{\link{BNN}}
#' @param optimiser an optimiser. These start with `opt.`.
#' See for example \code{\link{opt.ADAM}}
#' @param batchsize batch size
#' @param epochs number of epochs to run for
#'
#' @return Returns a vector. Use \code{\link{posterior_predictive}}
#' to obtain a prediction using this MAP estimate.
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
#'   find_mode(bnn, opt.RMSProp(), 10, 100)
#' }
#'
#' @export
find_mode <- function(bnn, optimiser, batchsize, epochs){
  juliacode <- sprintf("find_mode(%s, %i, %i, FluxModeFinder(%s, %s); showprogress = true)",
                       bnn$juliavar, batchsize, epochs, bnn$juliavar, optimiser$juliavar)
  return(JuliaCall::julia_eval(juliacode))
}
