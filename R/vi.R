
#' Use Bayes By Backprop to find Variational Approximation to BNN.
#'
#' This was proposed in Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra,
#' D. (2015, June). Weight uncertainty in neural network. In International
#' conference on machine learning (pp. 1613-1622). PMLR.
#'
#'@param bnn a BNN obtained using \code{\link{BNN}}
#'@param batchsize batch size
#'@param epochs number of epochs to run for
#'@param mc_samples samples to use in each iteration for the MC approximation
#'usually one is enough.
#'@param opt An optimiser. These all start with `opt.`. See for example \code{\link{opt.ADAM}}
#'@param n_samples_convergence At the end of each iteration convergence is checked using this
#'many MC samples.
#'
#'@return a list containing
#'\itemize{
#'    \item `juliavar` - julia variable storing VI
#'    \item `juliacode` - julia representation of function call
#'    \item `params` - variational family parameters for each iteration
#'    \item `losses` - BBB loss in each iteration
#'}
#' @examples
#' \dontrun{
#'   ## Needs previous call to `BayesFluxR_setup` which is time
#'   ## consuming and requires Julia and BayesFlux.jl
#'   BayesFluxR_setup(installJulia=TRUE, seed=123)
#'   net <- Chain(RNN(5, 1))
#'   like <- likelihood.seqtoone_normal(net, Gamma(2.0, 0.5))
#'   prior <- prior.gaussian(net, 0.5)
#'   init <- initialise.allsame(Normal(0, 0.5), like, prior)
#'   data <- matrix(rnorm(10*1000), ncol = 10)
#'   # Choosing sequences of length 10 and predicting one period ahead
#'   tensor <- tensor_embed_mat(data, 10+1)
#'   x <- tensor[1:10, , , drop = FALSE]
#'   # Last value in each sequence is the target value
#'   y <- tensor[11,,]
#'   bnn <- BNN(x, y, like, prior, init)
#'   vi <- bayes_by_backprop(bnn, 100, 100)
#'   vi_samples <- vi.get_samples(vi, n = 1000)
#' }
#'
#'@export
bayes_by_backprop <- function(bnn, batchsize, epochs,
                              mc_samples = 1,
                              opt = opt.ADAM(),
                              n_samples_convergence = 10){

  juliacode <- sprintf("bbb(%s, %i, %i; mc_samples = %i, opt = %s, n_samples_convergence = %i)",
                       bnn$juliavar, batchsize, epochs,
                       mc_samples, opt$juliavar,
                       n_samples_convergence)

  juliavar <- get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = %s;",
                                   juliavar, juliacode))


  params <- JuliaCall::julia_eval(sprintf("%s[2]", juliavar))
  losses <- JuliaCall::julia_eval(sprintf("%s[3]", juliavar))

  out <- list(juliavar = juliavar,
              juliacode = juliacode,
              params = params,
              losses = losses)

  return(out)
}

#' Draw samples form a variational family.
#'
#' @param vi obtained using \code{\link{bayes_by_backprop}}
#' @param n number of samples
#'
#' @return a matrix whose columns are draws from the variational posterior
#' @examples
#' \dontrun{
#'   ## Needs previous call to `BayesFluxR_setup` which is time
#'   ## consuming and requires Julia and BayesFlux.jl
#'   BayesFluxR_setup(installJulia=TRUE, seed=123)
#'   net <- Chain(RNN(5, 1))
#'   like <- likelihood.seqtoone_normal(net, Gamma(2.0, 0.5))
#'   prior <- prior.gaussian(net, 0.5)
#'   init <- initialise.allsame(Normal(0, 0.5), like, prior)
#'   data <- matrix(rnorm(10*1000), ncol = 10)
#'   # Choosing sequences of length 10 and predicting one period ahead
#'   tensor <- tensor_embed_mat(data, 10+1)
#'   x <- tensor[1:10, , , drop = FALSE]
#'   # Last value in each sequence is the target value
#'   y <- tensor[11,,]
#'   bnn <- BNN(x, y, like, prior, init)
#'   vi <- bayes_by_backprop(bnn, 100, 100)
#'   vi_samples <- vi.get_samples(vi, n = 1000)
#'   pp <- posterior_predictive(bnn, vi_samples)
#' }
#'
#' @export
vi.get_samples <- function(vi, n = 1){
  samples <- JuliaCall::julia_eval(sprintf("rand(%s[1], %i)", vi$juliavar, n))
  return(samples)
}
