
#' Stochastic Gradient Langevin Dynamics as proposed in Welling, M., & Teh, Y. W.
#' (n.d.). Bayesian Learning via Stochastic Gradient Langevin Dynamics. 8.
#'
#' Stepsizes will be adapted according to
#' \deqn{a(b+t)^{-\gamma}}
#'
#' @param stepsize_a See eq. above
#' @param stepsize_b See eq. above
#' @param stepsize_gamma see eq. above
#' @param min_stepsize Do not decrease stepsize beyond this
#'
#' @return a list with `juliavar`, `juliacode`, and all given arguments
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
#'   sampler <- sampler.SGLD()
#'   ch <- mcmc(bnn, 10, 1000, sampler)
#' }
#'
#' @export
sampler.SGLD <- function(stepsize_a = 0.1, stepsize_b = 0,
                         stepsize_gamma = 0.55, min_stepsize = -Inf){

  JuliaCall::julia_source(system.file("Julia/ascii-translate.jl", package = "BayesFluxR"))

  juliavar <- get_random_symbol()
  juliacode <- sprintf("ascii_SGLD(; stepsize_a = Float32(%e), stepsize_b = Float32(%e), stepsize_gamma = Float32(%e), min_stepsize = Float32(%s))",
                       stepsize_a, stepsize_b, stepsize_gamma, min_stepsize)
  JuliaCall::julia_command(sprintf("%s = %s;",
                                   juliavar, juliacode))
  out <- list(juliavar = juliavar, juliacode = juliacode,
              stepsize_a = stepsize_a, stepsize_b = stepsize_b,
              stepsize_gamma = stepsize_gamma)
  return(out)
}


#' Adaptive Metropolis Hastings as introduced in
#'
#' Haario, H., Saksman, E., & Tamminen, J. (2001). An adaptive Metropolis
#' algorithm. Bernoulli, 223-242.
#'
#' @param bnn BNN obtained using \code{\link{BNN}}
#' @param t0 Number of iterators before covariance adaptation will be started.
#' Also the lookback period for covariance adaptation.
#' @param sd Tuning parameter; See paper
#' @param eps Used for numerical reasons. Increase this if pos-def-error thrown.
#'
#' @return a list with `juliavar`, `juliacode`, and all given arguments
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
#'   sampler <- sampler.AdaptiveMH(bnn, 10, 1)
#'   ch <- mcmc(bnn, 10, 1000, sampler)
#' }
#'
#' @export
sampler.AdaptiveMH <- function(bnn, t0, sd, eps=1e-6){
  C0 = diag(x = rep(1, BNN.totparams(bnn)))
  sym.C0 = get_random_symbol()
  JuliaCall::julia_assign(sym.C0, C0)
  JuliaCall::julia_command(sprintf("%s = Float32.(%s);",
                                   sym.C0, sym.C0))

  juliavar <- get_random_symbol()
  juliacode <- sprintf("AdaptiveMH(%s, %i, Float32(%e), Float32(%e))",
                       sym.C0, t0, sd, eps)
  JuliaCall::julia_command(sprintf("%s = %s;",
                                   juliavar, juliacode))

  out <- list(juliavar = juliavar, juliacode = juliacode,
              C0 = C0, t0 = t0, sd = sd, eps = eps)
  return(out)
}

#' Gradient Guided Monte Carlo
#'
#' Proposed in Garriga-Alonso, A., & Fortuin, V. (2021). Exact langevin dynamics
#' with stochastic gradients. arXiv preprint arXiv:2102.01691.
#'
#' @param beta See paper
#' @param l stepsize
#' @param sadapter Stepsize adapter; Not used in original paper
#' @param madapter Mass adapter; Not used in ogirinal paper
#' @param steps Number of steps before accept/reject
#'
#' @return a list with `juliavar`, `juliacode` and all provided arguments.
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
sampler.GGMC <- function(beta = 0.1, l = 1.0,
                         sadapter = sadapter.DualAverage(1000),
                         madapter = madapter.FixedMassMatrix(),
                         steps = 3){

  JuliaCall::julia_source(system.file("Julia/ascii-translate.jl", package = "BayesFluxR"))

  juliavar <- get_random_symbol()
  juliacode <- sprintf("ascii_GGMC(; beta = Float32(%e), l = Float32(%e), sadapter = %s, madapter = %s, steps = %i)",
                       beta, l, sadapter$juliavar, madapter$juliavar, steps)

  JuliaCall::julia_command(sprintf("%s = %s;",
                                   juliavar, juliacode))

  out <- list(juliavar = juliavar,
              juliacode = juliacode,
              beta = beta,
              l = l,
              sadapter = sadapter,
              madapter = madapter,
              steps = steps)
  return(out)
}


#' Standard Hamiltonian Monte Carlo (Hybrid Monte Carlo).
#'
#' Allows for the use of stochastic gradients, but the validity of doing so is not clear.
#'
#' This is motivated by parts of the discussion in
#' Neal, R. M. (1996). Bayesian Learning for Neural Networks (Vol. 118). Springer
#' New York. https://doi.org/10.1007/978-1-4612-0745-0
#'
#' @param l stepsize
#' @param path_len number of leapfrog steps
#' @param sadapter Stepsize adapter
#' @param madapter Mass adapter
#'
#' @return a list with `juliavar`, `juliacode`, and all given arguments
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
#'   sampler <- sampler.HMC(1e-3, 3, sadapter = sadapter)
#'   ch <- mcmc(bnn, 10, 1000, sampler)
#' }
#'
#' @export
sampler.HMC <- function(l, path_len,
                        sadapter = sadapter.DualAverage(1000),
                        madapter = madapter.FixedMassMatrix()){
  juliavar <- get_random_symbol()
  juliacode <- sprintf("HMC(Float32(%e), %i; sadapter = %s, madapter = %s)",
                       l, path_len, sadapter$juliavar, madapter$juliavar)
  JuliaCall::julia_command(sprintf("%s = %s;",
                           juliavar, juliacode))

  out <- list(juliavar = juliavar,
              juliacode = juliacode,
              l = l,
              path_len = path_len,
              sadapter = sadapter,
              madapter = madapter)
  return(out)
}

#' Stochastic Gradient Nose-Hoover Thermostat as proposed in
#'
#' Proposed in Leimkuhler, B., & Shang, X. (2016). Adaptive thermostats for noisy
#' gradient systems. SIAM Journal on Scientific Computing, 38(2), A712-A736.
#'
#' This is similar to SGNHT as proposed in
#' Ding, N., Fang, Y., Babbush, R., Chen, C., Skeel, R. D., & Neven, H. (2014).
#' Bayesian sampling using stochastic gradient thermostats. Advances in neural
#' information processing systems, 27.
#'
#' @param l Stepsize
#' @param sigmaA Diffusion factor
#' @param xi Thermostat
#' @param mu Free parameter of thermostat
#' @param madapter Mass Adapter; Not used in original paper and thus
#' has no theoretical backing
#'
#' @return a list with `juliavar`, `juliacode` and all arguments provided
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
#'   sampler <- sampler.SGNHTS(1e-3)
#'   ch <- mcmc(bnn, 10, 1000, sampler)
#' }
#'
#' @export
sampler.SGNHTS <- function(l, sigmaA = 1, xi = 1, mu = 1,
                           madapter = madapter.FixedMassMatrix()){

  JuliaCall::julia_source(system.file("Julia/ascii-translate.jl", package = "BayesFluxR"))

  juliavar <- get_random_symbol()
  juliacode <- sprintf("ascii_SGNHTS(Float32(%e), Float32(%e); xi = Float32(%e), mu = Float32(%e), madapter = %s)",
                       l, sigmaA, xi, mu, madapter$juliavar)

  JuliaCall::julia_command(sprintf("%s = %s;",
                                   juliavar, juliacode))

  out <- list(juliavar = juliavar,
              juliacode = juliacode,
              l = l,
              sigmaA = sigmaA,
              xi = xi,
              mu = mu,
              madapter = madapter)

  return(out)
}

#' Sample from a BNN using MCMC
#'
#' @param bnn A BNN obtained using \code{\link{BNN}}
#' @param batchsize batchsize to use; Most samplers allow for batching.
#' For some, theoretical justifications are missing (HMC)
#' @param numsamples Number of mcmc samples
#' @param sampler Sampler to use; See for example \code{\link{sampler.SGLD}} and
#' all other samplers start with `sampler.` and are thus easy to identity.
#' @param continue_sampling Do not start new sampling, but rather continue sampling
#' For this, numsamples must be greater than the already sampled number.
#' @param start_value Values to start from. By default these will be
#' sampled using the initialiser in `bnn`.
#'
#' @return a list containing the `samples` and the `sampler` used.
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
#'   sampler <- sampler.SGNHTS(1e-3)
#'   ch <- mcmc(bnn, 10, 1000, sampler)
#' }
#'
#' @export
mcmc <- function(bnn, batchsize, numsamples,
                 sampler = sampler.SGLD(stepsize_a = 1.0),
                 continue_sampling = FALSE,
                 start_value = NULL){

  continue_sampling <- ifelse(continue_sampling, "true", "false")

  juliacode <- sprintf("ascii_mcmc(%s, %i, %i, %s; continue_sampling = %s)",
                       bnn$juliavar, batchsize, numsamples,
                       sampler$juliavar,
                       continue_sampling)


  if (!is.null(start_value)){
    sym.start_value <- get_random_symbol()
    JuliaCall::julia_assign(sym.start_value, start_value)
    JuliaCall::julia_command(sprintf("%s = Float32.(%s);",
                                     sym.start_value, sym.start_value))
    juliacode <- sprintf("ascii_mcmc(%s, %i, %i, %s; continue_sampling = %s, start_value = %s)",
                         bnn$juliavar, batchsize, numsamples,
                         sampler$juliavar,
                         continue_sampling, sym.start_value)
  }

  samples <- JuliaCall::julia_eval(sprintf("%s",juliacode))


  return(list(samples = samples, sampler = sampler))
}







