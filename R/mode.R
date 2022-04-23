
#' Find the posterior mode of a BNN using full gradients
#'
#' @param bnn BNN created using \code{\link{BNN}}
#' @param maxiter maximum iterations
#' @param tol Stop when update is less than tol percent
#'
#' @return A list containing
#' \itemize{
#'     \item has_converged - Whether the algorithm has converged
#'     \item mode - The mode (in vector form)
#' }
#'
#' @export
find_mode <- function(bnn, maxiter, tol=1e-6){
  sym.mode <- get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = find_mode(%s, %i, %f);",
                                   sym.mode,
                                   bnn$juliavar,
                                   maxiter,
                                   tol))
  converged <- JuliaCall::julia_eval(sprintf("%s[2]", sym.mode))
  mode <- JuliaCall::julia_eval(sprintf("%s[1]", sym.mode))
  out <- list(has_converged = converged,
              mode = mode)
  return(out)
}

#' Find the posterior mode of a BNN using Stochastic Gradient Descent
#'
#' @inheritParams find_mode
#' @param batchsize Mini-batchsize
#' @param init Vector of intial values; See \code{\link{BNN.totparams}}
#'
#' @return A list containing the following
#' \itemize{
#'     \item has_converged - Whether the algorithm has converged
#'     \item mode - The mode (in vector form)
#'     \item loss - Value of the logposterior at each point along the optimisation
#' }
#'
#' @export
find_mode_sgd <- function(bnn, batchsize, init, maxiter, tol=1e-6){
  sym.init <- get_random_symbol()
  JuliaCall::julia_assign(sym.init, init)
  sym.mode <- get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = find_mode_sgd(%s, %i, %s, %i, %f);",
                                   sym.mode,
                                   bnn$juliavar,
                                   batchsize,
                                   sym.init,
                                   maxiter,
                                   tol))
  converged <- JuliaCall::julia_eval(sprintf("%s[2]", sym.mode))
  mode <- JuliaCall::julia_eval(sprintf("%s[1]", sym.mode))
  loss <- JuliaCall::julia_eval(sprintf("%s[3]", sym.mode))

  out <- list(has_converged = converged,
              mode = mode,
              loss = loss)
  return(out)
}
