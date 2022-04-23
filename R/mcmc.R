
#' Use Stochastic Gradient Longevin Dynamics
#'
#' Uses Stochastic Gradient Longevin Dynamics according to
#' Welling, M., & Teh, Y. W. (2011).
#' Bayesian learning via stochastic gradient Langevin dynamics.
#' In Proceedings of the 28th international conference on machine learning (ICML-11) (pp. 681-688).
#'
#' @param bnn BNN formed using \code{\link{BNN}}
#' @param batchsize Mini-batchsize
#' @param maxiter Number of epochs to run SGLD for
#' @param init inital values; If `nchains==1` then a numeric vector
#'             of length `BNN.totparams(bnn)`, else a list of
#'             numeric vectors of the size above.
#' @param nchains Number of chains to run in parallel.
#'
#' @return If `nchains == 1` then return a matrix of dimension
#'         parameter x draws. Otherwise it returns a tensor of dimension
#'         parameter x draws x chain
#'
#' @export
sgld <- function(bnn, batchsize, maxiter,
                 init = NULL, nchains = 1){
  sym.init <- get_random_symbol()
  if (nchains == 1){
    if (is.null(init)) init <- rnorm(BNN.totparams(bnn))
    JuliaCall::julia_assign(sym.init, init)
    juliacode <- sprintf("sgld(%s, %i, %s, %i)",
                         bnn$juliavar,
                         batchsize,
                         sym.init,
                         maxiter)
    return(JuliaCall::julia_eval(juliacode))
  }
  if (is.null(init)) init <- lapply(1:nchains, function(x) rnorm(BNN.totparams(bnn)))
  JuliaCall::julia_assign(sym.init, init)
  JuliaCall::julia_command(sprintf("%s = [Float64.(init) for init in %s];",
                                   sym.init, sym.init))
  juliacode <- sprintf("sgld(%s, %i, %s, %i, %i)",
                       bnn$juliavar,
                       batchsize,
                       sym.init,
                       maxiter,
                       nchains)
  return(JuliaCall::julia_eval(sprintf("cat(%s...; dims = 3);", juliacode)))
}


#' Use Gradient Guided Monte Carlo with Stochastic Gradients
#'
#' Uses GGMC with Stochastic Gradients as proposed in
#' Garriga-Alonso, A., & Fortuin, V. (2021).
#' Exact langevin dynamics with stochastic gradients. arXiv preprint arXiv:2102.01691.
#'
#' @param bnn BNN formed using \code{\link{BNN}}
#' @param batchsize Mini-Batchsize
#' @param maxiter Number of epochs to run ggmc for
#' @param init see \code{\link{sgld}}
#' @param nchains Number of chains
#' @param l Step length
#' @param beta Momentum
#' @param keep_every If greater than 1, then delayed acceptance will be used.
#'                   Only the last draw will be kept.
#' @param adapruns How many runs to use for adaptation
#' @param kappa Decrease exponent for step size adaptation
#' @param goal_accept_rate Target acceptance rate
#' @param adaptM Adapt the mass matrix? Often fails currently
#' @param adapth Adapt h (This directly corresponds to adapting `l`)
#'
#' @return A list with the following content
#' \itemize{
#'     \item samples - A matrix of dimension parameters x draws if `nchains==1`, otherwise
#'           a tensor of dimensions parameters x draws x chains
#'     \item hastings - MH ratios.
#'     \item momenta - Momentum draws
#'     \item accept - accepted draw = 1
#' }
#'
#' @export
ggmc <- function(bnn, batchsize, maxiter, init = NULL, nchains = 1,
                 l = 0.01, beta = 0.5, keep_every = 10,
                 adapruns = 5000, kappa = 0.5, goal_accept_rate = 0.65,
                 adaptM = FALSE, adapth = TRUE) {
  sym.init <- get_random_symbol()
  sym.ggmc <- get_random_symbol()
  if (nchains == 1){
    if (is.null(init)) init <- rnorm(BNN.totparams(bnn))
    JuliaCall::julia_assign(sym.init, init)
    juliacode <- sprintf("%s = ggmc(%s, %i, %s, %i; l = %f, beta = %f, keep_every = %i, adapruns = %i, kappa = %f, goal_accept_rate = %f, adaptM = %s, adapth = %s);",
                         sym.ggmc, bnn$juliavar, batchsize, sym.init, maxiter,
                         l, beta, keep_every,
                         adapruns, kappa,
                         goal_accept_rate,
                         ifelse(adaptM, "true", "false"),
                         ifelse(adapth, "true", "false"))
  } else{
    if (is.null(init)) init <- lapply(1:nchains, function(x) rnorm(BNN.totparams(bnn)))
    JuliaCall::julia_assign(sym.init, init)
    JuliaCall::julia_command(sprintf("%s = [Float64.(init) for init in %s];",
                                     sym.init, sym.init))
    juliacode <- sprintf("%s = ggmc(%s, %i, %s, %i, %i; l = %f, beta = %f, keep_every = %i, adapruns = %i, kappa = %f, goal_accept_rate = %f, adaptM = %s, adapth = %s);",
                         sym.ggmc, bnn$juliavar, batchsize, sym.init, maxiter, nchains,
                         l, beta, keep_every,
                         adapruns, kappa,
                         goal_accept_rate,
                         ifelse(adaptM, "true", "false"),
                         ifelse(adapth, "true", "false"))

  }

  JuliaCall::julia_command(juliacode)
  samples <- JuliaCall::julia_eval(sprintf("%s[1]", sym.ggmc))
  hastings <- JuliaCall::julia_eval(sprintf("%s[2]", sym.ggmc))
  momenta <- JuliaCall::julia_eval(sprintf("%s[3]", sym.ggmc))
  accept <- JuliaCall::julia_eval(sprintf("%s[4]", sym.ggmc))
  out <- list(samples = samples,
              hastings = hastings,
              momenta = momenta,
              accept = accept)
  return(out)
}


