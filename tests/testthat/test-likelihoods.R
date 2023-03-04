
testthat::test_that("Feedforward Normal Likelihood", {
  BayesFluxR_setup(env_path = ".", nthreads = 3, pkg_check = FALSE)
  ch <- Chain(Dense(1, 1))
  ffnormal <- likelihood.feedforward_normal(ch, Gamma(2.0, 0.5))
  x <- JuliaCall::julia_eval(ffnormal$juliavar)
  expect_s3_class(x, "JuliaObject")
})


testthat::test_that("Feedforward TDist Likelihood", {
  BayesFluxR_setup(env_path = ".", nthreads = 3, pkg_check = FALSE)
  ch <- Chain(Dense(1, 1))
  ffnormal <- likelihood.feedforward_tdist(ch, Gamma(2.0, 0.5), 5.0)
  x <- JuliaCall::julia_eval(ffnormal$juliavar)
  expect_s3_class(x, "JuliaObject")
})

testthat::test_that("Seq-to-One Normal Likelihood", {
  BayesFluxR_setup(env_path = ".", nthreads = 3, pkg_check = FALSE)
  ch <- Chain(RNN(1, 1))
  ffnormal <- likelihood.seqtoone_normal(ch, Gamma(2.0, 0.5))
  x <- JuliaCall::julia_eval(ffnormal$juliavar)
  expect_s3_class(x, "JuliaObject")
})

testthat::test_that("Seq-to-One TDist Likelihood", {
  BayesFluxR_setup(env_path = ".", nthreads = 3, pkg_check = FALSE)
  ch <- Chain(RNN(1, 1))
  ffnormal <- likelihood.seqtoone_tdist(ch, Gamma(2.0, 0.5), 5.0)
  x <- JuliaCall::julia_eval(ffnormal$juliavar)
  expect_s3_class(x, "JuliaObject")
})

