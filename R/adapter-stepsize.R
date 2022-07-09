
#' Use a constant stepsize in mcmc
#'
#' @param l stepsize
#'
#' @return list with `juliavar`, `juliacode` and the given arguments
#'
#' @export
sadapter.Const <- function(l){
  juliavar <- get_random_symbol()
  juliacode <- sprintf("ConstantStepsize(%ff0)", l)
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
#'
#' @export
sadapter.DualAverage <- function(adapt_steps, initial_stepsize=1.0,
                                 target_accept = 0.65,
                                 gamma = 0.05, t0 = 10, kappa = 0.75) {

  juliavar <- get_random_symbol()
  juliacode <- sprintf("DualAveragingStepSize(%ff0; target_accept = %ff0,
                       gamma = %ff0, t0 = %i, kappa = %ff0, adapt_steps = %i)",
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



