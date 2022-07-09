function prior_predict_feedforward(bnn)
  predict(net) = vec(net(bnn.x))
  return predict
end

function prior_predict_seqtoone(bnn)
  predict(net) = vec([net(xx) for xx in eachslice(bnn.x; dims = 1)][end])
  return predict
end
