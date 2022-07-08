
#' Installs Julia packages if needed
#'
#' @param ... strings of package names
#'
.install_pkg <- function(...){
  for (pkg in as.character(list(...))) {
    JuliaCall::julia_install_package_if_needed(pkg)
  }
}


#' Obtain the status of the current Julia project
.julia_project_status <- function(){
  JuliaCall::julia_command("Pkg.status()")
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
#'
#' @export
.set_seed <- function(seed){
  JuliaCall::julia_command(sprintf("Random.seed!(%i);", seed))
  set.seed(seed)
  message("Set the seed of Julia and R to ", seed)
}

#' Set up of the Julia environment needed for BFlux
#'
#' This will set up a new Julia environment in the current working
#' directory or another folder if provided. This environment will
#' then be set with all Julia dependencies needed.
#'
#' @param pkg_check (Default=TRUE) Check whether needed Julia packages
#'                  are installed
#' @param nthreads (Default=4) How many threads to make available to Julia
#' @param seed Seed to be used.
#' @param env_path The path to were the Julia environment should be created.
#'                 By default, this is the current working directory.
#' @param ... Other parameters passed on to \code{\link[JuliaCall]{julia_setup}}
#'
#' @export
BFluxR_setup <- function(pkg_check = TRUE, nthreads = 4, seed = NULL, env_path = getwd(), ...){

  Sys.setenv(JULIA_NUM_THREADS = sprintf("%i", nthreads))
  julia <- JuliaCall::julia_setup(installJulia = TRUE, ...)
  JuliaCall::julia_library("Pkg")
  sym.env <- get_random_symbol()
  JuliaCall::julia_assign(sym.env, env_path)
  JuliaCall::julia_command(sprintf("Pkg.activate(%s)", sym.env))
  pkgs_needed <- list("https://github.com/enweg/BFlux.git", "Flux@0.13.0", "Distributions", "Random")
  if (pkg_check){
    do.call(.install_pkg, pkgs_needed)
  }
  do.call(.using, c(c("BFlux", "Flux"), pkgs_needed[-c(1:2)]))
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
  julia_layer_strings <- c()
  for (elem in list(...)){
    # julia <- paste0(julia, elem$julia, ",")
    julia_layer_strings <- c(julia_layer_strings, elem$julia)
  }
  julia_layer_strings <- paste0(julia_layer_strings, collapse = ", ")
  julia <- paste0(julia, julia_layer_strings)
  julia <- paste0(julia, ")")
  sym.net <- get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = %s", sym.net, julia))

  # creating a NetworkConstructor
  sym.nc <- get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = destruct(%s)",
                                   sym.nc, sym.net))

  out <- list(juliavar = sym.net, specification = julia, nc = sym.nc)
  return(out)
}

#' Create a Bayesian Neural Network
#'
#' @param x For a Feedforward structure, this must be a matrix of dimensions
#' variables x observations; For a recurrent structure, this must be a
#' tensor of dimensions sequence_length x number_variables x number_sequences;
#' In general, the last dimension is always the dimension over which will be batched.
#' @param y A vector or matrix with observations.
#' @param like Likelihood; See for example \code{\link{likelihood.feedforward_normal}}
#' @param prior Prior; See for example \code{\link{prior.gaussian}}
#' @param init Initialiser; See for example \code{\link{initialise.allsame}}
#'
#' @return List with the following content
#' \itemize{
#'     \item `juliavar` - the julia variable containing the BNN
#'     \item `juliacode` - the string representation of the BNN
#' }
#'
#' @export
BNN <- function(x, y, like, prior, init){
  sym.x <- get_random_symbol()
  sym.y <- get_random_symbol()

  JuliaCall::julia_assign(sym.x, x)
  JuliaCall::julia_command(sprintf("%s = Float32.(%s)",
                                   sym.x, sym.x))

  JuliaCall::julia_assign(sym.y, y)
  JuliaCall::julia_command(sprintf("%s = Float32.(%s)",
                                   sym.y, sym.y))


  juliavar <- get_random_symbol()
  juliacode <- sprintf("BNN(%s, %s, %s, %s, %s)",
                       sym.x, sym.y, like$juliavar,
                       prior$juliavar, init$juliavar)
  JuliaCall::julia_command(sprintf("%s = %s",
                           juliavar, juliacode))

  out <- list(juliavar = juliavar, juliacode = juliacode)
  return(out)
}

#' Obtain the total parameters of the BNN
#'
#' @param bnn A BNN formed using \code{\link{BNN}}
#' @export
BNN.totparams <- function(bnn) {
  totparams <- JuliaCall::julia_eval(sprintf("%s.num_total_params", bnn$juliavar))
  return(totparams)
}



