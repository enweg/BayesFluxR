
#' Use a constant stepsize in mcmc
#'
#' @param l stepsize
#'
#' @return list with `juliavar`, `juliacode` and the given arguments
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
#'   sadapter <- sadapter.Const(1e-5)
#'   sampler <- sampler.GGMC(sadapter = sadapter)
#'   ch <- mcmc(bnn, 10, 1000, sampler)
#' }
#'
#' @export
sadapter.Const <- function(l){
  juliavar <- get_random_symbol()
  juliacode <- sprintf("ConstantStepsize(Float32(%e))", l)
  JuliaCall::julia_command(sprintf("%s = %s",
                                   juliavar, juliacode))

  out <- list(juliavar = juliavar, juliacode = juliacode, l = l)
  return(out)
}


#' Use Dual Averaging like in STAN to tune stepsize
#'
#' @param adapt_steps number of adaptation steps
#' @param initial_stepsize initial stepsize
#' @param target_accept target acceptance ratio
#' @param gamma See STAN manual NUTS paper
#' @param t0 See STAN manual or NUTS paper
#' @param kappa See STAN manual or NUTS paper
#'
#' @return list with `juliavar`, `juliacode`, and all given arguments
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
#'   sadapter <- sadapter.DualAverage(100)
#'   sampler <- sampler.GGMC(sadapter = sadapter)
#'   ch <- mcmc(bnn, 10, 1000, sampler)
#' }
#'
#' @export
sadapter.DualAverage <- function(adapt_steps, initial_stepsize=1.0,
                                 target_accept = 0.65,
                                 gamma = 0.05, t0 = 10, kappa = 0.75) {

  juliavar <- get_random_symbol()
  juliacode <- sprintf("DualAveragingStepSize(Float32(%e); target_accept = Float32(%e),
                       gamma = Float32(%e), t0 = %i, kappa = Float32(%e), adapt_steps = %i)",
                       initial_stepsize, target_accept, gamma, t0,
                       kappa, adapt_steps)

  JuliaCall::julia_command(sprintf("%s = %s",
                                   juliavar, juliacode))

  out <- list(juliavar = juliavar,
              juliacode = juliacode,
              adapt_steps = adapt_steps,
              initial_stepsize = initial_stepsize,
              target_accept = target_accept,
              gamma = gamma,
              t0 = t0,
              kappa = kappa)
  return(out)
}



