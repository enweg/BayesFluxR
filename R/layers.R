
#' Create a Dense layer with `in_size` inputs and `out_size` outputs using `act` activation function
#'
#' @param in_size Input size
#' @param out_size Output size
#' @param act Activation function
#'
#' @return A list with the following content
#' \itemize{
#'      \item in_size - Input Size
#'      \item out_size - Output Size
#'      \item activation - Activation Function
#'      \item julia - Julia code representing the Layer
#' }
#'
#' @export
#'
Dense <- function(in_size, out_size, act = c("identity", "sigmoid", "tanh", "relu")) {
  act <- match.arg(act)
  juliacode <- sprintf("Dense(%i, %i, %s)", in_size, out_size, act)
  out <- list(
    in_size = in_size,
    out_size = out_size,
    activation = act,
    julia = juliacode
  )
  return(out)
}

#' Create a RNN layer with `in_size` input, `out_size` hidden state and `act` activation function
#'
#' @inheritParams Dense
#' @seealso \code{\link{Dense}}
#' @export
#'
RNN <- function(in_size, out_size, act = c("sigmoid", "tanh", "identity", "relu")){
  act <- match.arg(act)
  juliacode <- sprintf("RNN(%i, %i, %s)", in_size, out_size, act)
  out <- list(
    in_size = in_size,
    out_size = out_size,
    activation = act,
    julia = juliacode
  )
}

#' Create an LSTM layer with `in_size` input size, and `out_size` hidden state size
#'
#' @inheritParams Dense
#' @seealso \code{\link{Dense}}
#' @export
#'
LSTM <- function(in_size, out_size){
  juliacode <- sprintf("LSTM(%i, %i)", in_size, out_size)
  out <- list(
    in_size = in_size,
    out_size = out_size,
    julia = juliacode
  )
  return(out)
}
