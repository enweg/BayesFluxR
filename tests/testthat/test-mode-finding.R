

testthat::test_that("Find Mode Feedforward", {
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

  opt <- opt.ADAM()
  mode <- find_mode(bnn, opt, 10, 10)
  expect_equal(length(mode), BNN.totparams(bnn))

  opt <- opt.Descent()
  mode <- find_mode(bnn, opt, 10, 10)
  expect_equal(length(mode), BNN.totparams(bnn))

  opt <- opt.RMSProp()
  mode <- find_mode(bnn, opt, 10, 10)
  expect_equal(length(mode), BNN.totparams(bnn))
})


testthat::test_that("Find Mode Seq-to-One", {
  # We will follow other packages such as diffeqr and skip
  # Julia related tests on CRAN
  testthat::skip_on_cran()
  # BayesFluxR_setup(installJulia = FALSE, env_path = ".", nthreads = 3, pkg_check = FALSE)
  test_setup(nthreads = 3, pkg_check = FALSE)

  y <- rnorm(500)
  tensor <- tensor_embed_mat(matrix(y, ncol = 1), len_seq = 10+1)
  y <- tensor[11, , ]
  x <- tensor[1:10, , , drop = FALSE]
  net <- Chain(RNN(1, 1))
  prior <- prior.gaussian(net, 0.5)
  like <- likelihood.seqtoone_normal(net, Gamma(2.0, 0.5))
  init <- initialise.allsame(Normal(0, 0.5), like, prior)
  bnn <- BNN(x, y, like, prior, init)

  opt <- opt.ADAM()
  mode <- find_mode(bnn, opt, 10, 10)
  expect_equal(length(mode), BNN.totparams(bnn))

  opt <- opt.Descent()
  mode <- find_mode(bnn, opt, 10, 10)
  expect_equal(length(mode), BNN.totparams(bnn))

  opt <- opt.RMSProp()
  mode <- find_mode(bnn, opt, 10, 10)
  expect_equal(length(mode), BNN.totparams(bnn))
})





