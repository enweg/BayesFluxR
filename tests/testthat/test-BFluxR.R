
testthat::test_that("Check Initialisation", {
  # We will follow other packages such as diffeqr and skip
  # Julia related tests on CRAN
  testthat::skip_on_cran()
  # BayesFluxR_setup(installJulia = FALSE, env_path = ".", nthreads = 3)
  test_setup(nthreads = 3)
  expect_s3_class(JuliaCall::julia_eval("BayesFlux"), "JuliaObject")
  expect_s3_class(JuliaCall::julia_eval("Flux"), "JuliaObject")
  expect_s3_class(JuliaCall::julia_eval("Distributions"), "JuliaObject")
  expect_s3_class(JuliaCall::julia_eval("Random"), "JuliaObject")
  # This currently has a problem. The problem is in JuliaCall though
  # and hence we stop testing it for now.
  # TODO: wait until issue has been fixed in JuliaCall and start testing again
  # expect_equal(JuliaCall::julia_eval("Threads.nthreads()"), 3)
})

testthat::test_that("Setting seed in R and Julia", {
  # We will follow other packages such as diffeqr and skip
  # Julia related tests on CRAN
  testthat::skip_on_cran()
  # BayesFluxR_setup(installJulia = FALSE, env_path = ".", nthreads = 3)
  test_setup(nthreads = 3)

  .set_seed(6150533)
  rand_R = runif(1)
  rand_Julia = JuliaCall::julia_eval("rand()")
  .set_seed(6150533)
  rand_R_2 = runif(1)
  rand_Julia_2 = JuliaCall::julia_eval("rand()")

  expect_equal(rand_R, rand_R_2)
  expect_equal(rand_Julia, rand_Julia_2)
})

testthat::test_that("Network Chains", {
  # We will follow other packages such as diffeqr and skip
  # Julia related tests on CRAN
  testthat::skip_on_cran()
  # BayesFluxR_setup(installJulia = FALSE, env_path = ".", nthreads = 3)
  test_setup(nthreads = 3)

  ch = Chain(Dense(1, 1))
  expect_equal(ch$specification, "Chain(Dense(1, 1, identity))")

  ch = Chain(Dense(1, 10, "tanh"))
  expect_equal(ch$specification, "Chain(Dense(1, 10, tanh))")

  ch = Chain(RNN(1, 10))
  expect_equal(ch$specification, "Chain(RNN(1, 10, sigmoid))")

  ch = Chain(RNN(1, 10, "tanh"))
  expect_equal(ch$specification, "Chain(RNN(1, 10, tanh))")

  ch = Chain(LSTM(20, 1))
  expect_equal(ch$specification, "Chain(LSTM(20, 1))")

  ch = Chain(RNN(1, 10, "tanh"), LSTM(10, 20), Dense(20, 5, "relu"))
  expect_equal(ch$specification, "Chain(RNN(1, 10, tanh), LSTM(10, 20), Dense(20, 5, relu))")
})


testthat::test_that("Create BNN", {
  # We will follow other packages such as diffeqr and skip
  # Julia related tests on CRAN
  testthat::skip_on_cran()
  # BayesFluxR_setup(installJulia = FALSE, env_path = ".", nthreads = 3)
  test_setup(nthreads = 3)
  x <- matrix(rnorm(100), nrow = 1)
  y <- rnorm(100)
  ch <- Chain(Dense(1, 1))
  prior <- prior.gaussian(ch, 1.0)
  like <- likelihood.feedforward_normal(ch, Gamma(2.0, 0.5))
  init <- initialise.allsame(Normal(0, 10), like, prior)

  bnn <- BNN(x, y, like, prior, init)
  num_total_params <- JuliaCall::julia_eval(sprintf("%s.num_total_params", bnn$juliavar))
  expect_equal(num_total_params, 3)


  num_total_params <- BNN.totparams(bnn)
  expect_equal(num_total_params, 3)
})
