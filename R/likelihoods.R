
######################################################################
#### Feedforward Lkelihoods ##########################################


#' Use a Normal likelihood for a Feedforward network
#'
#' This creates a likelihood of the form
#' \deqn{y_i \sim Normal(net(x_i), \sigma)\;\forall i=1,...,N}
#'  where the \eqn{x_i} is fed through the network in a standard feedforward way.
#'
#' @param chain Network structure obtained using \code{link{Chain}}
#' @param sig_prior A prior distribution for sigma defined using
#'                  \code{\link{Gamma}}, \code{link{InverGamma}},
#'                  \code{\link{Truncated}}, \code{\link{Normal}}
#'
#' @return A list containing the following
#' \itemize{
#'     \item juliavar - julia variable containing the likelihood
#'     \item juliacode - julia code used to create the likelihood
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
#'   BNN.totparams(bnn)
#' }
#'
#' @export
likelihood.feedforward_normal <- function(chain, sig_prior){
  juliacode <- sprintf("FeedforwardNormal(%s, %s)",
                       chain$nc, sig_prior$juliavar)
  symbol <- get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = %s;", symbol, juliacode))
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
#' @examples
#' \dontrun{
#'   ## Needs previous call to `BayesFluxR_setup` which is time
#'   ## consuming and requires Julia and BayesFlux.jl
#'   BayesFluxR_setup(installJulia=TRUE, seed=123)
#'   net <- Chain(Dense(5, 1))
#'   like <- likelihood.feedforward_tdist(net, Gamma(2.0, 0.5), nu=8)
#'   prior <- prior.gaussian(net, 0.5)
#'   init <- initialise.allsame(Normal(0, 0.5), like, prior)
#'   x <- matrix(rnorm(5*100), nrow = 5)
#'   y <- rnorm(100)
#'   bnn <- BNN(x, y, like, prior, init)
#'   BNN.totparams(bnn)
#' }
#'
#' @export
likelihood.feedforward_tdist <- function(chain, sig_prior, nu=30){
  juliacode <- sprintf("BayesFlux.FeedforwardTDist(%s, %s, %ff0)",
                       chain$nc, sig_prior$juliavar, nu)
  symbol <- get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = %s;", symbol, juliacode))
  out <- list(juliavar = symbol, juliacode = juliacode)
  return(out)
}

######################################################################
#### Seq. To One #####################################################

#' Use a Normal likelihood for a seq-to-one recurrent network
#'
#' This creates a likelihood of the form
#' \deqn{y_i \sim Normal(net(x_i), \sigma), i=1,...,N}
#' Here \eqn{x_i} is a subsequence which will be fed through the recurrent
#' network to obtain the final output \eqn{net(x_i) = \hat{y}_i}. Thus, if
#' one has a single time series, and splits the single time series into subsequences
#' of length K which are then used to predict the next output of the time series, then
#' each \eqn{x_i} consists of K consecutive observations of the time series. In a sense
#' one constraints the maximum memory length of the network this way.
#'
#' @inheritParams likelihood.feedforward_normal
#' @return see \code{\link{likelihood.feedforward_normal}}
#' @examples
#' \dontrun{
#'   ## Needs previous call to `BayesFluxR_setup` which is time
#'   ## consuming and requires Julia and BayesFlux.jl
#'   BayesFluxR_setup(installJulia=TRUE, seed=123)
#'   net <- Chain(RNN(5, 1))
#'   like <- likelihood.seqtoone_normal(net, Gamma(2.0, 0.5))
#'   prior <- prior.gaussian(net, 0.5)
#'   init <- initialise.allsame(Normal(0, 0.5), like, prior)
#'   x <- array(rnorm(5*100*10), dim=c(10,5,100))
#'   y <- rnorm(100)
#'   bnn <- BNN(x, y, like, prior, init)
#'   BNN.totparams(bnn)
#' }
#'
#' @export
likelihood.seqtoone_normal <- function(chain, sig_prior){
  juliacode <- sprintf("BayesFlux.SeqToOneNormal(%s, %s)",
                       chain$nc, sig_prior$juliavar)
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
#' @examples
#' \dontrun{
#'   ## Needs previous call to `BayesFluxR_setup` which is time
#'   ## consuming and requires Julia and BayesFlux.jl
#'   BayesFluxR_setup(installJulia=TRUE, seed=123)
#'   net <- Chain(RNN(5, 1))
#'   like <- likelihood.seqtoone_tdist(net, Gamma(2.0, 0.5), nu=5)
#'   prior <- prior.gaussian(net, 0.5)
#'   init <- initialise.allsame(Normal(0, 0.5), like, prior)
#'   x <- array(rnorm(5*100*10), dim=c(10,5,100))
#'   y <- rnorm(100)
#'   bnn <- BNN(x, y, like, prior, init)
#'   BNN.totparams(bnn)
#' }
#'
#' @export
likelihood.seqtoone_tdist <- function(chain, sig_prior, nu = 30){
  juliacode <- sprintf("BayesFlux.SeqToOneTDist(%s, %s, %ff0)",
                       chain$nc, sig_prior$juliavar, nu)
  sym <- get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = %s;", sym, juliacode))
  out <- list(juliavar = sym, juliacode = juliacode)
  return(out)
}
