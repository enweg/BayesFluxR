
testthat::test_that("MCMC Sampling", {
  # We will follow other packages such as diffeqr and skip
  # Julia related tests on CRAN
  testthat::skip_on_cran()
  # BayesFluxR_setup(installJulia = FALSE, env_path = ".", nthreads = 3, pkg_check = FALSE)
  test_setup(nthreads = 3, pkg_check = FALSE)

  y <- rnorm(100)
  x <- matrix(rnorm(100), nrow = 1)
  net <- Chain(Dense(1, 1))
  prior <- prior.gaussian(net, 0.5)
  like <- likelihood.feedforward_normal(net, Gamma(2.0, 0.5))
  init <- initialise.allsame(Normal(0, 0.5), like, prior)
  bnn <- BNN(x, y, like, prior, init)


  sampler <- sampler.SGLD(stepsize_a = 1.0)

  ch <- mcmc(bnn, 10, 1000, sampler = sampler)
  expect_equal(dim(ch$samples)[2], 1000)
  expect_equal(dim(ch$samples)[1], BNN.totparams(bnn))
  ch2 <- mcmc(bnn, 10, 2000, sampler = sampler, continue_sampling = TRUE)
  expect_equal(dim(ch2$samples)[2], 2000)
  expect_true(all.equal(ch$samples, ch2$samples[, 1:1000]))

  sampler <- sampler.SGLD(stepsize_a = 1.0)
  ch <- mcmc(bnn, 10, 1000, sampler = sampler, start_value = rep(10, BNN.totparams(bnn)))
  expect_equal(dim(ch$samples)[2], 1000)
  expect_equal(dim(ch$samples)[1], BNN.totparams(bnn))
  expect_true(ch$samples[1, 1] > 1)
})
