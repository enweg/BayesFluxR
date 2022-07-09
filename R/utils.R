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
