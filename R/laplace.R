

#' Draw values from a distribution
#'
#' @param dist Should either be a prior distribution (see for example
#'             \code{\link{Gamma}}) or a laplace approximation.
#' @param n Number of draws
#'
#' @export
draw <- function(dist, n){
  JuliaCall::julia_eval(sprintf("rand(%s, %i)", dist$juliavar, n))
}


#' Form a laplace approximation of the posterior
#'
#' Uses `M` mode estimates to form a laplace approximation around
#' each of these modes. The returned distribution is then a mixture
#' of normals with equal weights. Currently only diagonal covarainces
#' are implemented.
#'
#' @param bnn A BNN formed using \code{\link{BNN}}
#' @param maxiter Maximum number of itertions to find modes
#' @param M Number of modes to find
#' @param tol tolerance (converged when subsequence changes are less than `tol` percent apart)
#'
#' @return A list containing
#' \itemize{
#'     \item has_converged - Boolean vector giving whether each mode estimate as converged
#'     \item juliavar - Julia variable holding the laplace approximation
#' }
#'
#' @export
laplace <- function(bnn, maxiter, M = 1, tol=1e-6){
  sym.laplace = get_random_symbol()
  juliacode <- sprintf("%s = laplace(%s, %i, %i, %f; diag = true);",
                       sym.laplace,
                       bnn$juliavar,
                       maxiter,
                       M,
                       tol)
  JuliaCall::julia_command(juliacode)
  converged = JuliaCall::julia_eval(sprintf("%s.c", sym.laplace))
  out <- list(has_converged = converged,
              juliavar = sym.laplace)
  return(out)
}


#' Use Sampling-Importance-Resampling to correct laplace approximations
#'
#' SIR is implemented using sampling without replacement. Thus, the inital sample
#' `n` should be much larger than the final sample size `k`
#'
#' @param bnn A BNN formed using \code{\link{BNN}}
#' @param lapprox A laplace approximation formed using \code{\link{laplace}}
#' @param n Inital numer of draws
#' @param k Final number of re-draws
#'
#' @return A Matrix; Rows are parameters, columns are draws
#'
#' @export
SIR.laplace <- function(bnn, lapprox, n, k){
  juliacode <- sprintf("SIR_laplace(%s, %s, %i, %i);",
                       bnn$juliavar, lapprox$juliavar,
                       n, k)
  JuliaCall::julia_eval(juliacode)
}


