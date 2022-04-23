#' Creates a random string that is used as variable in julia
get_random_symbol <- function() paste0(sample(letters, 5, replace = TRUE), collapse = "")
