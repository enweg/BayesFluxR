#' Create a Gamma Prior
#'
#' Creates a Gamma prior in Julia using Distributions.jl
#'
#' @param shape shape parameter
#' @param scale scale parameter
#'
#' @return A list with the following content
#' \itemize{
#'     \item juliavar - julia variable containing the distribution
#'     \item juliacode - julia code used to create the distribution
#' }
#' @examples
#' \dontrun{
#'   ## Needs previous call to `BayesFluxR_setup` which is time
#'   ## consuming and requires Julia and BayesFlux.jl
#'   BayesFluxR_setup(installJulia=TRUE, seed=123)
#'   net <- Chain(Dense(5, 1))
#'   like <- likelihood.feedforward_normal(net, Gamma(2.0, 0.5))
#' }
#'
#' @export
Gamma <- function(shape=2.0, scale=2.0){
  juliacode <- sprintf("Gamma(%f, %f)", shape, scale)
  symbol <- get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = %s;", symbol, juliacode))
  out <- list(juliavar = symbol, juliacode = juliacode)
  return(out)
}

#' Create an Inverse-Gamma Prior
#'
#' Creates and Inverse Gamma prior in Julia using Distributions.jl
#'
#' @inheritParams Gamma
#' @seealso \code{\link{Gamma}}
#'
#' @return A list with the following content
#' \itemize{
#'     \item juliavar - julia variable containing the distribution
#'     \item juliacode - julia code used to create the distribution
#' }
#' @examples
#' \dontrun{
#'   ## Needs previous call to `BayesFluxR_setup` which is time
#'   ## consuming and requires Julia and BayesFlux.jl
#'   BayesFluxR_setup(installJulia=TRUE, seed=123)
#'   net <- Chain(Dense(5, 1))
#'   like <- likelihood.feedforward_normal(net, InverseGamma(2.0, 0.5))
#' }
#' @export
InverseGamma <- function(shape=2.0, scale=2.0){
  juliacode <- sprintf("InverseGamma(%f, %f)", shape, scale)
  symbol <- get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = %s;", symbol, juliacode))
  out <- list(juliavar = symbol, juliacode = juliacode)
  return(out)
}

#' Create a Normal Prior
#'
#' Creates a Normal prior in Julia using Distributions.jl. This can
#' then be truncated using \code{\link{Truncated}} to obtain a prior
#' that could then be used as a variance prior.
#'
#' @param mu Mean
#' @param sigma Standard Deviation
#'
#' @return see \code{\link{Gamma}}
#' @examples
#' \dontrun{
#'   ## Needs previous call to `BayesFluxR_setup` which is time
#'   ## consuming and requires Julia and BayesFlux.jl
#'   BayesFluxR_setup(installJulia=TRUE, seed=123)
#'   net <- Chain(Dense(5, 1))
#'   like <- likelihood.feedforward_normal(net, Truncated(Normal(0, 0.5), 0, Inf))
#' }
#'
#' @export
Normal <- function(mu=0, sigma=1){
  juliacode <- sprintf("Normal(%f, %f)", mu, sigma)
  symbol <- get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = %s;", symbol, juliacode))
  out <- list(juliavar = symbol, juliacode = juliacode)
  return(out)
}

#' Truncates a Distribution
#'
#' Truncates a Julia Distribution between `lower` and `upper`.
#'
#' @param dist A Julia Distribution created using \code{\link{Gamma}},
#'             \code{\link{InverseGamma}} ...
#' @param lower lower bound
#' @param upper upper bound
#'
#' @return see \code{\link{Gamma}}
#' @examples
#' \dontrun{
#'   ## Needs previous call to `BayesFluxR_setup` which is time
#'   ## consuming and requires Julia and BayesFlux.jl
#'   BayesFluxR_setup(installJulia=TRUE, seed=123)
#'   net <- Chain(Dense(5, 1))
#'   like <- likelihood.feedforward_normal(net, Truncated(Normal(0, 0.5), 0, Inf))
#' }
#'
#' @export
Truncated <- function(dist, lower, upper){
  juliacode <- sprintf("Truncated(%s, %f, %f)", dist$juliacode, lower, upper)
  symbol <- get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = %s;", symbol, juliacode))
  out <- list(juliavar = symbol, juliacode = juliacode)
  return(out)
}
