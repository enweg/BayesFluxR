
#' Use the diagonal of sample covariance matrix as inverse mass matrix.
#'
#' @param adapt_steps Number of adaptation steps
#' @param windowlength Lookback window length for calculation of covariance
#' @param kappa How much to shrink towards the identity
#' @param epsilon Small value to add to diagonal so as to avoid numerical
#' non-pos-def problem
#'
#' @return list containing `juliavar` and `juliacode` and all given arguments.
#'
#' @export
madapter.DiagCov <- function(adapt_steps, windowlength,
                             kappa = 0.5, epsilon = 1e-6){

  juliavar <- get_random_symbol()
  juliacode <- sprintf("DiagCovMassAdapter(%i, %i; kappa = %ff0, epsilon = %ff0)",
                       adapt_steps, windowlength, kappa, epsilon)
  JuliaCall::julia_command(sprintf("%s = %s",
                                   juliavar, juliacode))

  out <- list(juliavar = juliavar, juliacode = juliacode,
              adapt_steps = adapt_steps, windowlength = windowlength,
              kappa = kappa, epsilon = epsilon)
  return(out)
}

#' Use a fixed mass matrix
#'
#' @param mat (Default=NULL); inverse mass matrix; If `NULL`, then
#' identity matrix will be used
#'
#' @return list with `juliavar` and `juliacode` and given matrix or `NULL`
#'
#' @export
madapter.FixedMassMatrix <- function(mat = NULL){
  juliacode <- "FixedMassAdapter()"
  if (!is.null(mat)){
    sym.mat <- get_random_symbol()
    JuliaCall::julia_assign(sym.mat, mat)
    JuliaCall::julia_command(sprintf("%s = Float32.(%s)",
                                     sym.mat, sym.mat))

    juliacode <- sprintf("FixedMassAdapter(%s)",
                         sym.mat)
  }

  juliavar <- get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = %s",
                                   juliavar, juliacode))

  out <- list(juliavar = juliavar,
              juliacode = juliacode,
              minv = mat)
  return(out)
}

#' Use the full covariance matrix as inverse mass matrix
#'
#' @inheritParams madapter.DiagCov
#'
#' @return see \code{\link{madapter.DiagCov}}
#'
#' @export
madapter.FullCov <- function(adapt_steps, windowlength,
                             kappa = 0.5, epsilon = 1e-6){

  juliavar <- get_random_symbol()
  juliacode <- sprintf("FullCovMassAdapter(%i, %i; kappa = %ff0, epsilon = %ff0)",
                       adapt_steps, windowlength, kappa, epsilon)
  JuliaCall::julia_command(sprintf("%s = %s",
                                   juliavar, juliacode))

  out <- list(juliavar = juliavar,
              juliacode = juliacode,
              adapt_steps = adapt_steps,
              windowlength = windowlength,
              kappa = kappa,
              epsilon = epsilon)
  return(out)
}


#' Use RMSProp to adapt the inverse mass matrix.
#'
#' Use RMSProp as a preconditions/mass matrix adapter. This was proposed in Li, C.,
#' Chen, C., Carlson, D., & Carin, L. (2016, February). Preconditioned stochastic
#' gradient Langevin dynamics for deep neural networks. In Thirtieth AAAI
#' Conference on Artificial Intelligence for the use in SGLD and related methods.
#'
#' @param adapt_steps number of adaptation steps
#' @param lambda see above paper
#' @param alpha see above paper
#'
#' @return list with `juliavar` and `juliacode` and all given arguments
#'
#' @export
madapter.RMSProp <- function(adapt_steps, lambda = 1e-5, alpha = 0.99){

  JuliaCall::julia_source(system.file("Julia/ascii-translate.jl", package = "BFluxR"))

  juliavar <- get_random_symbol()
  juliacode <- sprintf("ascii_RMSPropMassAdapter(%i; lambda = %ff0, alpha = %ff0)",
                       adapt_steps, lambda, alpha)

  JuliaCall::julia_command(sprintf("%s = %s",
                                   juliavar, juliacode))

  out <- list(juliavar = juliavar,
              juliacode = juliacode,
              lambda = lambda,
              alpha = alpha)
  return(out)
}






