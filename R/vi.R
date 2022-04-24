
#' Draw from ADVI distribution
#'
#' @inheritParams draw.vi
#'
#' @export
draw.vi.advi <- function(vi, n){
  if (!class(vi) == "advi") stop("vi must be of class advi")
  JuliaCall::julia_eval(sprintf("rand(%s, %i)", vi$juliavar, n))
}

#' Draw from BBB distribution
#'
#' @inheritParams draw.vi
#'
#' @export
draw.vi.bbb <- function(vi, n){
  JuliaCall::julia_eval(sprintf("rand(%s, %i)",
                                vi$juliadist, n))
}

#' Draws from a variational distribution
#'
#' @param vi A variational distribution
#' @param n Number of draws
#'
#' @return A Matrix; Rows are parameters, columns are draws;
#'
#' @export
draw.vi <- function(vi, n){
  UseMethod("draw.vi")
}

#' Use Automatic Differentiation Variational Inference (ADVI)
#'
#' Uses ADVI to approximate the BNN posterior using a diagonal MvNormal
#'
#' @param bnn A BNN formed using \code{\link{BNN}}
#' @param samples_per_step Samples used in each iteration to
#'                         estimate ELBO
#' @param maxiter Iterations to run AFVI for
#'
#' @return A object of class advi from which can be drawn using
#'         \code{\link{draw.vi}}
#'
#' @export
advi <- function(bnn, samples_per_step, maxiter){
  sym.advi <- get_random_symbol()
  juliacode <- sprintf("%s = advi(%s, %i, %i);",
                       sym.advi,
                       bnn$juliavar,
                       samples_per_step,
                       maxiter)
  JuliaCall::julia_command(juliacode)
  out <- list(juliavar = sym.advi)
  class(out) <- "advi"
  return(out)
}

#' Use Bayes By Backprop (BBB)
#'
#' Uses BBB to form a variational approximation to the posterior using
#' a diagonal multivariate normal distribution.
#'
#' @param bnn A BNN formed using \code{\link{BNN}}
#' @param samples_per_step How many samples to take in each iteration (to approximate ELBO)
#' @param maxiter Maximum number of iterations to run the algorithm
#' @param batchsize Mini-Batchsize to use
#' @param nchains How many BBB approximations to form
#'
#' @return A objevt of class bbb from which can be drawn using \code{\link{draw.vi}}
#'
#' @export
bbb <- function(bnn, samples_per_step, maxiter, batchsize, nchains = 1){
  sym.bbb <- get_random_symbol()
  if (nchains == 1) {
    juliacode <- sprintf("%s = bbb(%s, %i, %i, %i);", sym.bbb,
                         bnn$juliavar, samples_per_step, maxiter, batchsize)
  } else {
    juliacode <- sprintf("%s = bbb(%s, %i, %i, %i, %i);", sym.bbb,
                         bnn$juliavar, samples_per_step, maxiter, batchsize, nchains)
  }
  JuliaCall::julia_command(juliacode)
  sym.dist <- get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = %s[1];",
                                   sym.dist, sym.bbb))
  losses <- JuliaCall::julia_eval(sprintf("%s[4]", sym.bbb))
  out <- list(juliavar = sym.bbb,
              juliadist = sym.dist,
              losses = losses)
  class(out) <- "bbb"
  return(out)
}
