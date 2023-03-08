
#' Initialises all parameters of the network, all hyper parameters
#' of the prior and all additional parameters
#' of the likelihood by drawing random values from `dist`.
#'
#' @param dist A distribution; See for example \code{\link{Normal}}
#' @param like A likelihood; See for example \code{\link{likelihood.feedforward_normal}}
#' @param prior A prior; See for example \code{\link{prior.gaussian}}
#'
#' @return A list containing the following
#' \itemize{
#'     \item `juliavar` - julia variable storing the initialiser
#'     \item `juliacode` - julia code used to create the initialiser
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
initialise.allsame <- function(dist, like, prior){
  juliacode <- sprintf("InitialiseAllSame(%s, %s, %s)",
                       dist$juliavar, like$juliavar, prior$juliavar)
  juliavar <- get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = %s;",
                                   juliavar, juliacode))
  out <- list(juliavar = juliavar, juliacode = juliacode)
  return(out)
}
