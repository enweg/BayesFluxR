#' Creates a random string that is used as variable in julia
get_random_symbol <- function() {
  nframe <- sys.nframe()
  caller <- ""
  if (nframe > 1){
    caller <- deparse(sys.calls()[[nframe-1]])
    caller <- strsplit(caller, "\\(")[[1]][1]
    caller <- gsub("\\.", "_", caller)
    caller <- paste0(caller, "_")
  }
  sym <- paste0(sample(letters, 5, replace = TRUE), collapse = "")
  paste0(caller, sym)
}


#' Embed a matrix of timeseries into a tensor
#'
#' This is used when working with recurrent networks, especially in
#' the case of seq-to-one modelling. Creates overlapping subsequences
#' of the data with length `len_seq`. Returned dimensions are seq_len x num_vars x num_subsequences.
#'
#' @param mat Matrix of time series
#' @param len_seq subsequence length
#'
#' @return A tensor of dimension: len_seq x num_vars x num_subsequences
#' @examples
#' \dontrun{
#'   ## Needs previous call to `BayesFluxR_setup` which is time
#'   ## consuming and requires Julia and BayesFlux.jl
#'   BayesFluxR_setup(installJulia=TRUE, seed=123)
#'   net <- Chain(RNN(5, 1))
#'   like <- likelihood.seqtoone_normal(net, Gamma(2.0, 0.5))
#'   prior <- prior.gaussian(net, 0.5)
#'   init <- initialise.allsame(Normal(0, 0.5), like, prior)
#'   data <- matrix(rnorm(5*1000), ncol = 5)
#'   # Choosing sequences of length 10 and predicting one period ahead
#'   tensor <- tensor_embed_mat(data, 10+1)
#'   x <- tensor[1:10, , , drop = FALSE]
#'   # Last value in each sequence is the target value
#'   y <- tensor[11,1,]
#'   bnn <- BNN(x, y, like, prior, init)
#'   BNN.totparams(bnn)
#' }
#'
#' @export
tensor_embed_mat <- function(mat, len_seq){
  num_sequences = nrow(mat) - len_seq + 1
  tensor <- array(rep(NA, len_seq*ncol(mat)*num_sequences),
                  dim = c(len_seq, ncol(mat), num_sequences))
  for (i in 1:num_sequences){
    tensor[, , i] <- mat[i:(i + len_seq - 1), ]
  }
  return(tensor)
}


#'
#' @param x some array
ndims <- function(x){
  if (is.array(x)) {
    return(length(dim(x)))
  }
  if (length(x) > 1) return(1)
  return(0)
}

#' Print a summary of a BNN
#'
#' @param object A BNN created using \code{\link{BNN}}
#' @param ... Not used
#' @export
summary.BNN <- function(object, ...){
  summary_n_independent <- NROW(object$x)
  cat("Number of independent variables:", summary_n_independent, "\n")
  summary_n_dependent <- NCOL(object$y)
  cat("Number of dependent variables:", summary_n_dependent, "\n")
  summary_n_observations <- NCOL(object$x)
  cat("Number of observations:", summary_n_observations, "\n")
  summary_like_name <- JuliaCall::julia_eval(sprintf("string(typeof(%s.like).name.name)", object$juliavar))
  cat("Likelihood:", summary_like_name, "\n")
  summary_prior_name <- JuliaCall::julia_eval(sprintf("string(typeof(%s.prior).name.name)", object$juliavar))
  cat("Prior:", summary_prior_name, "\n")
  summary_init_name <- JuliaCall::julia_eval(sprintf("string(typeof(%s.init).name.name)", object$juliavar))
  cat("Initalisation method:", summary_init_name, "\n")
  summary_n_network_params <- JuliaCall::julia_eval(sprintf("%s.like.nc.num_params_network", object$juliavar))
  cat("Number of network parameters:", summary_n_network_params, "\n")
  summary_n_total_params <- BNN.totparams(object)
  cat("Number of total parameters:", summary_n_total_params, "\n")
  code <- sprintf('"Chain(\n\t$(join(%s.like.nc(randn(Float32, %s.like.nc.num_params_network)).layers, ",\n\t"))\n)"',
                  object$juliavar, object$juliavar)
  summary_network_structure <- JuliaCall::julia_eval(code)
  cat("Network Structure:\n", summary_network_structure, "\n")
}

#' Convert draws array to conform with `bayesplot`
#'
#' BayesFluxR returns draws in a matrix of dimension
#' params x draws. This cannot be used with the `bayesplot` package
#' which expects an array of dimensions draws x  chains x params.
#'
#' @param ch Chain of draws obtained using \code{\link{mcmc}}
#' @param param_names If `NULL`, the parameter names will be of the
#' form `param_1`, `param_2`, etc. If `param_names` is a string,
#' the parameter names will start with the string with the number
#' of the parameter attached to it. If `param_names` is a vector, it
#' has to provide a name for each paramter in the chain.
#' @returns Returns an array of dimensions draws x chains x params.
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
#'   ch <- to_bayesplot(ch)
#'   library(bayesplot)
#'   mcmc_intervals(ch, pars = paste0("param_", 1:10))
#' }
#' @export
to_bayesplot <- function(ch, param_names = NULL) {
  if (is.null(param_names)) {
    param_names <- paste0("param_", 1:dim(ch)[1])
  } else if (length(param_names) == 1) {
    param_names <- paste0(param_names, 1:dim(ch)[1])
  }
  ch_array <- array(ch, dim = c(dim(ch), 1),
                    dimnames = list("params" = param_names,
                                    "iterations" = NULL,
                                    "chains" = c("chain:1")))
  ch_array <- aperm(ch_array, c(2, 3, 1))
  return(ch_array)
}
