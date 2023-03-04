ascii_SGLD(; stepsize_a = 1.0f0, stepsize_b = 0f0, stepsize_gamma = 0.55f0, min_stepsize = Float32(-Inf)) = BayesFlux.SGLD(; stepsize_a = stepsize_a, stepsize_b = stepsize_b, stepsize_γ = stepsize_gamma, min_stepsize = min_stepsize)

ascii_RMSPropMassAdapter(adapt_steps; lambda = 1f-5, alpha = 0.99f0) = RMSPropMassAdapter(adapt_steps; λ = lambda, α = alpha)

ascii_GGMC(; beta = 0.55f0, l = 0.0001f0, sadapter = DualAveragingStepSize(l), madapter = FixedMassAdapter(), steps = 1) = GGMC(; β = beta, l = l, sadapter = sadapter, madapter = madapter, steps = steps)

ascii_SGNHTS(l, sigmaA = 1f0; xi = 1f0, mu = 1f0, madapter = FixedMassAdapter()) = SGNHTS(l, sigmaA; xi = xi, μ = mu, madapter = madapter)

ascii_mcmc(bnn, batchsize, numsamples, sampler; continue_sampling = false, start_value = vcat(bnn.init()...)) = BayesFlux.mcmc(bnn, batchsize, numsamples, sampler; continue_sampling = continue_sampling, θstart = start_value)