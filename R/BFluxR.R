
#' Installs Julia packages if needed
#'
#' @param ... strings of package names
#'
.install_pkg <- function(...){
  for (pkg in as.character(list(...))) {
    JuliaCall::julia_install_package_if_needed(pkg)
  }
}

#' Loads Julia packages
#'
#' @param ... strings of package names
.using <- function(...){
  for (pkg in list(...)) {
    JuliaCall::julia_library(pkg)
  }
}

#' Set a seed both in Julia and R
#'
#' @param seed seed to be used
.set_seed <- function(seed){
  JuliaCall::julia_command(sprintf("Random.seed!(%i)", seed))
  set.seed(seed)
  message("Set the seed of Julia and R to ", seed)
}

#' Set up of the Julia environment needed for BFlux
#'
#' @param pkg_check (Default=TRUE) Check whether needed Julia packages
#'                  are installed
#' @param nthreads (Default=4) How many threads to make available to Julia
#' @param seed Seed to be used.
#' @param ... Other parameters passed on to \code{\link[JuliaCall]{julia_setup}}
#'
#' @export
BFluxR_setup <- function(pkg_check = TRUE, nthreads = 4, seed = NULL, ...){

  Sys.setenv(JULIA_NUM_THREADS = sprintf("%i", nthreads))
  julia <- JuliaCall::julia_setup(installJulia = TRUE, ...)
  # pkgs_needed <- list("git@github.com:enweg/BFlux.git", "Flux", "Distributions", "Random")
  pkgs_needed <- list("git@github.com:enweg/BFlux.git", "Flux@0.13.0", "Distributions", "Random")
  if (pkg_check){
    do.call(.install_pkg, pkgs_needed)
  }
  do.call(.using, c(list("BFlux", "Flux"), pkgs_needed[-c(1:2)]))
  if (!is.null(seed)) .set_seed(seed)
}

#' Chain various layers together to form a network
#'
#' @param ... Comma separated layers
#'
#' @return List with the following content
#' \itemize{
#'     \item juliavar - the julia variable containing the network
#'     \item specification - the string representation of the network
#' }
#'
#' @export
Chain <- function(...){
  julia <- "Chain("
  for (elem in list(...)){
    julia <- paste0(julia, elem$julia, ",")
  }
  julia <- paste0(julia, ")")
  sym.net <- get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = %s", sym.net, julia))
  out <- list(juliavar = sym.net, specification = julia)
  return(out)
}

#' Create A Bayesian Neural Network using BFlux.jl
#'
#' @param net A network created using \code{\link{Chain}}
#' @param loglike A likelihood created using, for example, \code{\link{likelihood.feedforward_normal}}
#' @param y outcomes
#' @param x features
#'
#' @return A list containing the following
#' \itemize{
#'     \item juliavar - julia variable containing the BNN
#'     \item juliacode - julia code used to create the BNN
#' }
#'
#' @export
#'
BNN <- function(net, loglike, y, x){
  sym.x <- get_random_symbol()
  sym.y <- get_random_symbol()
  JuliaCall::julia_assign(sym.x, x)
  if (ndims(x) == 3){
    # Recurrent Case. Transform Tensor to Vector of Matrix
    JuliaCall::julia_command(sprintf("%s = BFlux.to_RNN_format(%s)",
                                     sym.x, sym.x))
  }
  JuliaCall::julia_assign(sym.y, y)
  juliacode <- sprintf("BNN(%s, %s, %s, %s)",
                       net$juliavar,
                       loglike$juliavar,
                       sym.y,
                       sym.x)
  sym.bnn <- get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = %s;",
                                   sym.bnn,
                                   juliacode))
  out <- list(juliavar = sym.bnn,
              juliacode = juliacode)
  return(out)
}

#' Obtain the total parameters of the BNN
#'
#' @param bnn A BNN formed using \code{\link{BNN}}
#' @export
BNN.totparams <- function(bnn) {
  totparams <- JuliaCall::julia_eval(sprintf("%s.totparams", bnn$juliavar))
  return(totparams)
}


#' Obtain posterior predictive draws
#'
#' @param bnn A BNN formed using \code{\link{BNN}}
#' @param samples Either a matrix of dimensions parameters x draws
#'                or a tensor of dimensions parameters x draws x chains
#' @param newx New X to use for posterior predictions
#'
#' @return Returns a matrix of dimensions N x draws if samples is a matrix
#'         and otherwise a tensor of dimensions N x draws x chains
#'
#' @export
posterior_predict <- function(bnn, samples, newx = NULL){
  sym.x = get_random_symbol()
  sym.samples = get_random_symbol()
  JuliaCall::julia_assign(sym.samples, samples)
  if (!is.null(newx)){
    JuliaCall::julia_assign(sym.x, newx)
    if (ndims(newx) == 3){
      JuliaCall::julia_command(sprintf("%s = BFlux.to_RNN_format(%s);",
                                       sym.x, sym.x))
    }
  } else {
    JuliaCall::julia_command(sprintf("%s = %s.x;", sym.x, bnn$juliavar))
  }

  JuliaCall::julia_eval(sprintf("posterior_predict(%s, %s; newx = %s)",
                                bnn$juliavar, sym.samples, sym.x))
}





