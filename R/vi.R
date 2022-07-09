

bayes_by_backprop <- function(bnn, batchsize, epochs,
                              mc_samples = 1,
                              opt = opt.ADAM(),
                              n_samples_convergence = 10){

  juliacode <- sprintf("bbb(%s, %i, %i; mc_samples = %i, opt = %s, n_samples_convergence = %i)",
                       bnn$juliavar, batchsize, epochs,
                       mc_samples, opt$juliavar,
                       n_samples_convergence)

  juliavar <- get_random_symbol()
  JuliaCall::julia_command(sprintf("%s = %s",
                                   juliavar, juliacode))


  params <- JuliaCall::julia_eval(sprintf("%s[2]", juliavar))
  losses <- JuliaCall::julia_eval(sprintf("%s[3]", juliavar))

  out <- list(juliavar = juliavar,
              juliacode = juliacode,
              params = params,
              losses = losses)

  return(out)
}


vi.get_samples <- function(vi, n = 1){
  samples <- JuliaCall::julia_eval(sprintf("rand(%s[1], %i)", vi$juliavar, n))
  return(samples)
}
