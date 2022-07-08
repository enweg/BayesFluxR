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

#' helper function
#'
#' @param y Observations
#' @param len_seq length of sequence
.tensor_embed <- function(y, len_seq){
  tensor <- c()
  n_seq <- length(y) - len_seq + 1
  for (i in 1:n_seq){
    tensor <- c(tensor, y[i:(i + n_seq - 1)])
  }
  tensor <- array(tensor, c(1, n_seq, len_seq))
  return(tensor)
}

#' Make a tensor of sequences
#'
#' Given sequences, these are usually feature sequences, tranform them
#' into a tensor of dimension features x num_sequences x len_sequence.
#'
#' @param ... sequences
#' @param len_seq the desired sequence length
#'
#' @examples
#' tensor_embed(1:5, len_seq = 2)
#' tensor_embed(1:5, 6:10, 11:15, len_seq = 2)
#'
#' @export
tensor_embed <- function(..., len_seq){
  tensor <- NULL
  for (feature_seq in list(...)){
    feature_tensor <- .tensor_embed(feature_seq, len_seq)
    tensor <- if (is.null(tensor)) feature_tensor else abind::abind(tensor, feature_tensor, along = 1)
  }
  return(tensor)
}

#' helper function to determine dimensions of object
#'
#' @param x some array
ndims <- function(x){
  if (is.array(x)) {
    return(length(dim(x)))
  }
  if (length(x) > 1) return(1)
  return(0)
}
