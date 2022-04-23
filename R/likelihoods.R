
######################################################################
#### Prior Distributions for Variance ################################

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
#'
#' @export
Gamma <- function(shape=2.0, scale=2.0){
  juliacode <- sprintf("Gamma(%f, %f)", shape, scale)
  symbol <- get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = %s", symbol, juliacode))
  out <- list(juliavar = symbol, juliacode = juliacode)
  return(out)
}

#' Create an InverGamma Prior
#'
#' Creates and Inverse Gamma prior in Julia using Distributions.jl
#'
#' @inheritParams Gamma
#' @seealso \code{\link{Gamma}}
#'
#' @export
InverseGamma <- function(shape=2.0, scale=2.0){
  juliacode <- sprintf("InverseGamma(%f, %f)", shape, scale)
  symbol <- get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = %s", symbol, juliacode))
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
#'
#' @export
Normal <- function(mu=0, sigma=1){
  juliacode <- sprintf("Normal(%f, %f)", mu, sigma)
  symbol <- get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = %s", symbol, juliacode))
  out <- list(juliavar = symbol, juliacode = juliacode)
  return(out)
}

#' Truncates a Distribution
#'
#' Truncates a Julia Disribution between `lower` and `upper`.
#'
#' @param dist A Julia Distribution created using \code{\link{Gamma}},
#'             \code{\link{InverseGamma}} ...
#' @param lower lower bound
#' @param upper upper bound
#'
#' @return see \code{\link{Gamma}}
#'
#' @export
Truncated <- function(dist, lower, upper){
  juliacode <- sprintf("Truncated(%s, %f, %f)", dist$juliacode, lower, upper)
  symbol <- get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = %s", symbol, juliacode))
  out <- list(juliavar = symbol, juliacode = juliacode)
  return(out)
}

######################################################################
#### Feedforward Lkelihoods ##########################################


#' Use a Normal likelihood for a Feedforward network
#'
#' This creates a likelihood of the form
#' \deqn{y_i \sim Normal(net(x_i), \sigma)\;\forall i=1,...,N}
#'  where the \eqn{x_i} is fed through the network in a standard feedforward way.
#'
#' @param sig_prior A prior distribution for sigma defined using
#'                  \code{\link{Gamma}}, \code{link{InverGamma}},
#'                  \code{\link{Truncated}}, \code{\link{Normal}}
#'
#' @return A list containing the following
#' \itemize{
#'     \item juliavar - julia variable containing the likelihood
#'     \item juliacode - julia code used to create the likelihood
#' }
#'
#' @export
likelihood.feedforward_normal <- function(sig_prior){
  juliacode <- sprintf("BFlux.FeedforwardNormal(%s, Float64)", sig_prior$juliavar)
  symbol <- get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = %s", symbol, juliacode))
  out <- list(juliavar = symbol, juliacode = juliacode)
  return(out)
}

#' Use  a t-Distribution likelihood for a Feedforward network
#'
#' This creates a likelihood of the form
#' \deqn{\frac{y_i - net(x_i)}{\sigma} \sim T_\nu\;\forall i=1,...,N}
#'  where the \eqn{x_i} is fed through the network in the standard feedforward way.
#'
#' @inheritParams likelihood.feedforward_normal
#' @param nu DF of TDist
#'
#' @return see \code{\link{likelihood.feedforward_normal}}
#'
#' @export
likelihood.feedforward_tdist <- function(sig_prior, nu=30){
  juliacode <- sprintf("BFlux.FeedforwardTDist(%s, %f)", sig_prior$juliavar, nu)
  symbol <- get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = %s", symbol, juliacode))
  out <- list(juliavar = symbol, juliacode = juliacode)
  return(out)
}

######################################################################
#### Seq. To One #####################################################

#' Use a Normal likelihood for a seq-to-one recurrent network
#'
#' This creates a likelihood of the form
#' \deqn{y_i \sim Normal(net(x_i), \sigma), i=1,...,N}
#' Here \eqn{x_i} is a sibsequence which will be fed through the recurrent
#' network to obtain the final output \eqn{net(x_i) = \hat{y}_i}. Thus, if
#' one has a single time series, and splits the single time series into subsequences
#' of length K which are then used to predict the next output of the time series, then
#' each \eqn{x_i} consists of K consecutive obsevations of the time series. In a sense
#' one constraints the maximum memory length of the network this way.
#'
#' @inheritParams likelihood.feedforward_normal
#' @return see \code{\link{likelihood.feedforward_normal}}
#'
#' @export
likelihood.seqtoone_normal <- function(sig_prior){
  juliacode <- sprintf("BFlux.SeqToOneNormal(%s, Float64)", sig_prior$juliavar)
  sym <- get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = %s;", sym, juliacode))
  out <- list(juliavar = sym, juliacode = juliacode)
  return(out)
}


#' Use a T-likelihood for a seq-to-one recurrent network.
#'
#' See \code{\link{likelihood.seqtoone_normal}} and \code{\link{likelihood.feedforward_tdist}}
#' for details,
#'
#' @inheritParams likelihood.feedforward_tdist
#'
#' @return see \code{\link{likelihood.feedforward_normal}}
#'
#' @export
likelihood.seqtoone_tdist <- function(sig_prior, nu = 30){
  juliacode <- sprintf("BFlux.SeqToOneTDist(%s, %f)",
                       sig_prior$juliavar, nu)
  sym <- get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = %s;", sym, juliacode))
  out <- list(juliavar = sym, juliacode = juliacode)
  return(out)
}
