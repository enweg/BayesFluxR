
testthat::test_that("Initialise All Same", {
  # We will follow other packages such as diffeqr and skip
  # Julia related tests on CRAN
  testthat::skip_on_cran()
  # BayesFluxR_setup(installJulia = FALSE, env_path = ".", nthreads = 3, pkg_check = FALSE)
  test_setup(nthreads = 3, pkg_check = FALSE)
  ch <- Chain(Dense(1, 1))
  prior <- prior.gaussian(ch, 1.0)
  like <- likelihood.feedforward_normal(ch, Gamma(2.0, 0.5))
  init <- initialise.allsame(Normal(0, 10), like, prior)

  x <- JuliaCall::julia_eval(init$juliavar)
  expect_s3_class(x, "JuliaObject")
})
